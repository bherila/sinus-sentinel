import Foundation
import Security
import SinusAppleFFI

/// `TokenProvider` implemented over Security.framework. Rust never sees the PHR
/// bearer token itself on Apple platforms — this Keychain item is the only copy.
///
/// `kSecClassGenericPassword` under service "SinusSentinel" / account
/// "phr-api-token": not arbitrary strings — that is exactly the (service,
/// account) pair the desktop tray's `keyring` crate uses, so a Mac that
/// already ran the tray app keeps its token when this shell reads the same
/// Keychain item.
///
/// Holds no state, so `Sendable` (required by the `TokenProvider` protocol)
/// is satisfied trivially — do not add a cache here. `ForeignTokenStore` on
/// the Rust side already caches per-process; a second cache here would be the
/// one that goes stale, since only `SyncController::setToken`/`clearToken`
/// know to invalidate it.
final class KeychainTokenProvider: TokenProvider {
    init() {}

    func getToken() throws -> String? {
        var query = Self.baseQuery
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        switch status {
        case errSecSuccess:
            guard let data = result as? Data, let token = String(data: data, encoding: .utf8) else {
                return nil
            }
            return token
        case errSecItemNotFound:
            return nil
        default:
            throw Self.error(for: status)
        }
    }

    func setToken(token: String) throws {
        let data = Data(token.utf8)
        // Set on the update as well as the add. An item this app created
        // already has it, but one inherited from the tray app's `keyring`
        // entry may not, and a token the sync thread cannot read after a
        // reboot fails in a way that looks like a server problem.
        let attributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
        ]
        let updateStatus = SecItemUpdate(Self.baseQuery as CFDictionary, attributes as CFDictionary)
        switch updateStatus {
        case errSecSuccess:
            return
        case errSecItemNotFound:
            var addQuery = Self.baseQuery
            // The sync thread flushes unattended, including right after a
            // reboot before anyone has unlocked the machine —
            // `WhenUnlocked` would make that flush fail. `ThisDeviceOnly`
            // keeps the token out of Keychain (iCloud) sync, since it is
            // meaningless anywhere but the machine whose device identity it
            // authenticates.
            addQuery.merge(attributes) { _, new in new }
            let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
            guard addStatus == errSecSuccess else {
                throw Self.error(for: addStatus)
            }
        default:
            throw Self.error(for: updateStatus)
        }
    }

    func clearToken() throws {
        let status = SecItemDelete(Self.baseQuery as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw Self.error(for: status)
        }
    }

    private static var baseQuery: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "SinusSentinel",
            kSecAttrAccount as String: "phr-api-token",
        ]
    }

    private static func error(for status: OSStatus) -> TokenError {
        // `SecCopyErrorMessageString` is macOS-only; iOS/tvOS/watchOS builds
        // of this same file fall back to the bare status code.
        #if os(macOS)
        if let description = SecCopyErrorMessageString(status, nil) as String? {
            return TokenError.Keychain(message: "OSStatus \(status): \(description)")
        }
        #endif
        return TokenError.Keychain(message: "OSStatus \(status)")
    }
}
