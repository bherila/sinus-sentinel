import Foundation

/// Normalizes what a user types into Settings › PHR's server URL field. Pure
/// — no engine, no I/O — so `SyncModel.setServerUrl` can validate before
/// saving instead of applying whatever was typed and finding out later, the
/// way the field used to (SPEC's motivating incident: the URL was correct,
/// but nothing told the user that, so they suspected it anyway).
enum ServerUrlNormalizer {
    /// Not `Result<String, String>`: `String` does not conform to `Error`, and
    /// wrapping it in one just to satisfy that would add a type nobody needs —
    /// this shape is exactly `Result`'s, `.success`/`.failure` included.
    enum Outcome: Equatable {
        case success(String)
        case failure(String)
    }

    /// A deep link into a specific patient's settings — e.g. copied from a
    /// browser tab already on the sinus-settings page. The Rust client
    /// already tolerates a full endpoint here (`SyncConfig::server_root`),
    /// but the field itself should show the root a second machine's client
    /// would actually construct, not the page the user happened to copy from.
    private static let deepLinkMarker = "/api/phr/patients/"

    /// `.success` carries the normalized string to store and display.
    /// Blank is itself a valid, normalized value — it means "sync off"; see
    /// `crates/app/src/settings.rs::server_url`'s doc comment, which gates
    /// sync on the patient id rather than on this being non-empty.
    static func normalize(_ raw: String) -> Outcome {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return .success("")
        }

        let rootOnly: String
        if let range = trimmed.range(of: deepLinkMarker) {
            rootOnly = String(trimmed[trimmed.startIndex..<range.lowerBound])
        } else {
            rootOnly = trimmed
        }

        let withScheme = rootOnly.contains("://") ? rootOnly : "https://\(rootOnly)"

        guard let url = URL(string: withScheme), let host = url.host, !host.isEmpty else {
            return .failure("That doesn't look like a server address.")
        }
        guard let scheme = url.scheme, scheme == "http" || scheme == "https" else {
            return .failure("The server address must start with http:// or https://.")
        }

        var normalized = withScheme
        while normalized.hasSuffix("/") {
            normalized.removeLast()
        }
        return .success(normalized)
    }
}
