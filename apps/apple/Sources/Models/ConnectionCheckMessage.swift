import SinusAppleFFI

/// Turns one `AppleConnectionCheck` into the exact copy Settings › PHR shows
/// for "Test connection". Pure — no engine, no SwiftUI — so it is testable
/// directly instead of only reachable through a live network round trip. Kept
/// in its own file, separate from `SyncModel`, the same way
/// `FeedbackMessageFormatter` is kept separate from `HistoryModel`/
/// `TrainingModel` — the test binary only wants the pure mapping, not
/// `SyncModel`'s `AppKit`/`SyncController` baggage.
enum ConnectionCheckMessage {
    static func message(for outcome: AppleConnectionCheck) -> String {
        switch outcome {
        case .ok:
            return "Connected — the server accepted the token."
        case .unauthenticated:
            return "The server rejected the stored token (expired, revoked, or wrong). Create a new API key in the PHR and save it here."
        case .patientNotFound:
            return "Connected and authenticated, but this patient id does not exist on that server."
        case .forbidden:
            return "Connected and authenticated, but the token does not grant access to this patient."
        case .noServerUrl:
            return "No server URL is set — add one above."
        case .noPatientId:
            return "No patient id is set — add one above."
        case .noToken:
            return "No API token is set — add one in the API token section above."
        case .offlineStrict:
            return "Sync mode is offline-strict, which never makes network calls. Switch modes above to test the connection."
        case .unreachable(let detail):
            return "Could not reach the server: \(detail)"
        case .http(let status):
            return "Unexpected server response (HTTP \(status))."
        }
    }

    /// Whether `message(for:)`'s text describes a failure — `.ok` is the only
    /// success. A structured check rather than a substring test on the
    /// rendered text, so wording changes above can never change which color
    /// a result renders in.
    static func isFailure(_ outcome: AppleConnectionCheck) -> Bool {
        if case .ok = outcome {
            return false
        }
        return true
    }
}
