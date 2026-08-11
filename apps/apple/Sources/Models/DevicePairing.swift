import Foundation
import CryptoKit

/// Pure pieces of "Sign in with PHR" — the device-pairing flow that replaces
/// pasting an API token by hand. No `AuthenticationServices`, no networking:
/// this file only builds requests and parses responses, so it is exercised
/// directly by `apple-test.sh`'s no-XCTest harness. The interactive half
/// (`ASWebAuthenticationSession`, the actual POST) lives in
/// `Platform/PhrSignInSession.swift`, which the test binary deliberately does
/// not link (see `scripts/apple-test.sh`'s curated source list).
///
/// Wire contract (frozen, shared with the Laravel PHR — do not change without
/// updating the server too):
///   1. Authorize: `{serverRoot}/device-pairing?device_id=...&name=...&code_challenge=...&redirect_uri=sinussentinel://paired`
///   2. Callback: `sinussentinel://paired?code=...` or `?error=denied`
///   3. Exchange: `POST {serverRoot}/api/device-pairing/exchange`
///      body `{"code":...,"code_verifier":...,"device_id":...}` ->
///      200 `{"token":...,"expires_at":...,"device_name":...}`, 400 invalid,
///      429 throttled.
enum DevicePairing {
    /// RFC 7636 PKCE verifiers may use a much larger character set, but the
    /// FROZEN contract only promises ASCII letters and digits, so that is all
    /// this generates.
    private static let verifierAlphabet = Array(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )
    private static let verifierLength = 96
    private static let callbackScheme = "sinussentinel"

    /// A built authorize request: the URL to open in the browser, and the
    /// verifier that produced its `code_challenge`. The exchange step needs
    /// the verifier back, so callers must carry both halves together rather
    /// than re-deriving one from the other (the challenge is one-way).
    struct PairingRequest: Equatable {
        let verifier: String
        let url: URL
    }

    /// Builds the authorize URL via `URLComponents` so every query value is
    /// properly percent-encoded. `randomVerifier` exists only so tests can
    /// inject a fixed verifier instead of `generateVerifier()`'s random one;
    /// production callers leave it `nil`. `nil` back means `serverRoot` was
    /// empty or could not be parsed as a URL at all.
    static func makeRequest(
        serverRoot: String,
        deviceId: String,
        deviceName: String,
        randomVerifier: String? = nil
    ) -> PairingRequest? {
        // `ServerUrlNormalizer` strips trailing slashes on save, but a URL
        // stored before it existed may still carry one — trim here too, or
        // the appended path becomes `//device-pairing` and 404s.
        var root = serverRoot
        while root.hasSuffix("/") {
            root.removeLast()
        }
        guard !root.isEmpty, var components = URLComponents(string: root) else {
            return nil
        }
        components.path += "/device-pairing"

        let verifier = randomVerifier ?? generateVerifier()
        components.queryItems = [
            URLQueryItem(name: "device_id", value: deviceId),
            URLQueryItem(name: "name", value: deviceName),
            URLQueryItem(name: "code_challenge", value: challenge(for: verifier)),
            URLQueryItem(name: "redirect_uri", value: "\(callbackScheme)://paired"),
        ]
        guard let url = components.url else { return nil }
        return PairingRequest(verifier: verifier, url: url)
    }

    /// `base64url(SHA-256(verifier))`, no padding — PKCE's `S256` transform,
    /// minus the padding characters the FROZEN contract says to strip.
    ///
    /// Derivation for the test vector `"test-verifier"` (computed here, not
    /// on-device, so the test can hardcode it as an independent check):
    ///   `echo -n test-verifier | shasum -a 256` ->
    ///   `2416e2a8e34658f6809b05e4ffc6d3e949e53dfae7eb90f7d9e665252fb3186d`
    ///   That hex, base64-encoded, is `JBbiqONGWPaAmwXk/8bT6UnlPfrn65D32eZlJS+zGG0=`;
    ///   swapping `+`/`/` for `-`/`_` and stripping the trailing `=` gives
    ///   `JBbiqONGWPaAmwXk_8bT6UnlPfrn65D32eZlJS-zGG0` — see
    ///   `DevicePairingTests.swift`.
    static func challenge(for verifier: String) -> String {
        let digest = SHA256.hash(data: Data(verifier.utf8))
        return Data(digest).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    /// 96 random characters from `verifierAlphabet`, drawn with the system's
    /// CSPRNG (`Array.randomElement` defaults to `SystemRandomNumberGenerator`)
    /// rather than `Int.random`'s seedable generator.
    static func generateVerifier() -> String {
        var generator = SystemRandomNumberGenerator()
        return String((0..<verifierLength).map { _ in verifierAlphabet.randomElement(using: &generator)! })
    }

    /// What `ASWebAuthenticationSession`'s callback URL means.
    enum CallbackOutcome: Equatable {
        case code(String)
        case denied
        case malformed
    }

    /// Distinguishes the three shapes `sinussentinel://paired` can arrive in:
    /// a one-time `code`, an explicit `error=denied`, or neither — anything
    /// else is `malformed` rather than guessed at.
    static func code(fromCallback url: URL) -> CallbackOutcome {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            return .malformed
        }
        let items = components.queryItems ?? []
        if let code = items.first(where: { $0.name == "code" })?.value, !code.isEmpty {
            return .code(code)
        }
        if items.first(where: { $0.name == "error" })?.value == "denied" {
            return .denied
        }
        return .malformed
    }

    /// JSON body for `POST {serverRoot}/api/device-pairing/exchange` —
    /// stable, snake_case keys, matching the FROZEN contract exactly.
    private struct ExchangeRequest: Encodable {
        let code: String
        let codeVerifier: String
        let deviceId: String

        enum CodingKeys: String, CodingKey {
            case code
            case codeVerifier = "code_verifier"
            case deviceId = "device_id"
        }
    }

    static func exchangeBody(code: String, verifier: String, deviceId: String) -> Data {
        let request = ExchangeRequest(code: code, codeVerifier: verifier, deviceId: deviceId)
        // Encoding this fixed, all-`String` shape cannot throw; `try!` keeps
        // callers from handling an error that has no reachable case.
        return try! JSONEncoder().encode(request)
    }

    /// The exchange's 200 body. A plain `Decodable` struct (rather than a
    /// manual keyed-container walk) already tolerates unknown extra fields —
    /// `JSONDecoder` only complains about *missing* required keys, not
    /// unrecognized ones.
    private struct ExchangeResponse: Decodable {
        let token: String
        let expiresAt: String
        let deviceName: String

        enum CodingKeys: String, CodingKey {
            case token
            case expiresAt = "expires_at"
            case deviceName = "device_name"
        }
    }

    struct ExchangeSuccess: Equatable {
        let token: String
        let expiresAt: String
        let deviceName: String
    }

    enum ExchangeOutcome: Equatable {
        case success(ExchangeSuccess)
        case invalidCode
        case throttled
        case serverError(String)
    }

    /// Maps the exchange's HTTP status/body onto `ExchangeOutcome`. 400 is a
    /// single, deliberately indistinguishable "invalid or expired" message
    /// per the FROZEN contract (it must not leak which); 422 (malformed
    /// shape) and anything else unrecognized both fall through to
    /// `.serverError`, same as a 200 whose body does not decode.
    static func parseExchangeResponse(status: Int, data: Data) -> ExchangeOutcome {
        switch status {
        case 200:
            guard let response = try? JSONDecoder().decode(ExchangeResponse.self, from: data) else {
                return .serverError("The server's response could not be understood.")
            }
            return .success(
                ExchangeSuccess(
                    token: response.token,
                    expiresAt: response.expiresAt,
                    deviceName: response.deviceName
                )
            )
        case 400:
            return .invalidCode
        case 429:
            return .throttled
        default:
            return .serverError("Unexpected server response (HTTP \(status)).")
        }
    }
}
