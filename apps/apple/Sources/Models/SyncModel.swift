import Foundation
import Observation
import SinusAppleFFI

#if os(macOS)
import AppKit
#endif
#if os(iOS)
import UIKit
#endif

/// Owns the `SyncController` and everything Settings › PHR renders. Mirrors
/// the desktop tray's `SettingsForm` PHR section (`app.rs:689-796`), including
/// its exact status strings, so the two shells read the same to a user who
/// runs both.
@MainActor
@Observable
final class SyncModel {
    private(set) var status: SyncStatusSnapshot
    private(set) var phr: PhrSettings?
    private(set) var tokenStatus: String
    private(set) var message: String?
    /// Result of the last live "Test connection", or `nil` before one has run.
    /// Distinct from `status.error`: that reports the *background* driver's
    /// last flush failure, this reports an explicit, on-demand probe of the
    /// server — the thing "Check token" never actually did (see
    /// `AppleConnectionCheck`'s doc for the incident).
    private(set) var connectionStatus: String?
    /// Whether `connectionStatus` describes a failure — kept as a structured
    /// flag rather than having the view sniff the text, so a wording change
    /// to `ConnectionCheckMessage` can never silently change which color a
    /// result renders in.
    private(set) var connectionFailed = false
    /// Set for the duration of `checkConnection()`'s blocking network call,
    /// so the view can disable the button rather than let a user queue up
    /// several probes against a server that has not answered the first one.
    private(set) var isCheckingConnection = false
    /// Set for the duration of `signIn()`'s browser round trip, so the view
    /// can disable the "Sign in with PHR…" button rather than let a second
    /// tap open a second `ASWebAuthenticationSession` on top of the first.
    private(set) var isSigningIn = false
    /// Set by `suspend()`, cleared by `resume()` — distinct from "no engine"
    /// (this device never got as far as starting sync at all): `resume()`
    /// only rebuilds the controller when this is true.
    private(set) var isSuspended = false

    /// Raised after each driver tick so projections containing per-row sync
    /// state can replace pending indicators with confirmed ones.
    var onStatusChanged: () -> Void = {}

    private var engine: AppleEngine?
    private var controller: SyncController?
    /// Kept so `resume()` can rebuild the controller without asking
    /// `EngineHost` to remember it on our behalf.
    private var tokens: TokenProvider?

    @ObservationIgnored
    private var terminationObserver: NSObjectProtocol?

    init() {
        status = SyncStatusSnapshot(
            state: .idle,
            mode: .autoBatch,
            pendingEvents: 0,
            pendingWork: 0,
            quiet: false,
            error: nil,
            lastSuccessEpochMs: nil
        )
        tokenStatus = "Token status not checked."
    }

    deinit {
        if let terminationObserver {
            NotificationCenter.default.removeObserver(terminationObserver)
        }
    }

    /// Builds the observer bridge and the driver thread, then seeds `status`
    /// and `phr` from it. Left for `EngineHost` to call after the engine
    /// exists; on failure `EngineHost` reports the error and leaves this
    /// model controller-less, so every write below simply no-ops — the same
    /// shape `MonitorModel`/`HistoryModel` use for a nil engine.
    func start(engine: AppleEngine, tokens: TokenProvider) throws {
        self.engine = engine
        self.tokens = tokens
        try buildController(engine: engine, tokens: tokens)

        #if os(macOS)
        // `Drop` on the Rust side only signals the driver thread to stop; it
        // does not join. Without an explicit, bounded shutdown here, quitting
        // mid-flush tears the thread down rather than giving it the chance
        // `SyncController::shutdown` exists to provide.
        terminationObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.shutdown()
            }
        }
        #endif
    }

    /// Builds the observer bridge and the driver thread, then seeds `status`
    /// and `phr` from it. Shared by `start` and `resume` — the only
    /// difference between a cold start and a resume after `suspend()` is who
    /// is calling.
    private func buildController(engine: AppleEngine, tokens: TokenProvider) throws {
        let observer = SyncStatusBridge { [weak self] status in
            Task { @MainActor in
                guard let self else { return }
                self.status = status
                self.onStatusChanged()
            }
        }
        let controller = try SyncController(engine: engine, tokens: tokens, observer: observer)
        self.controller = controller
        status = controller.status()
        reload()
    }

    func shutdown() {
        controller?.shutdown(timeoutMs: 3000)
    }

    /// Called when the app backgrounds with no active monitoring session
    /// keeping the process alive. The point is the driver thread, not the
    /// database: a flush blocks on a socket for up to the HTTP client's
    /// 30-second timeout, and a process suspended mid-flush resumes holding a
    /// connection the network moved on from. `shutdown` is the bounded way to
    /// stop and wait; releasing the last Swift reference afterward is what
    /// then closes the third `Store` connection `SyncController::new` opened,
    /// so a rebuild in `resume()` does not stack a second one beside it.
    func suspend() {
        controller?.shutdown(timeoutMs: 3000)
        controller = nil
        isSuspended = true
    }

    /// Rebuilds the controller torn down by `suspend()`. A no-op unless
    /// actually suspended; on failure this sets `message` rather than
    /// throwing — offline is a legitimate steady state everywhere else in
    /// this file, and the same holds coming back from the background.
    func resume() {
        guard isSuspended, let engine, let tokens else { return }
        do {
            try buildController(engine: engine, tokens: tokens)
            isSuspended = false
        } catch {
            message = "Could not resume sync: \(error.localizedDescription)"
        }
    }

    func reload() {
        guard let engine else { return }
        phr = try? engine.phrSettings()
    }

    /// Validates and normalizes `raw` before saving — SPEC's motivating
    /// incident is half "the token was invalid" and half "the field applied
    /// silently on focus-loss with no feedback at all", so a rejected URL now
    /// sets `message` instead of being written verbatim, and an accepted one
    /// gets a confirmation plus an immediate live check rather than leaving
    /// the user to wonder whether it actually works.
    func setServerUrl(_ raw: String) {
        guard let engine else { return }
        switch ServerUrlNormalizer.normalize(raw) {
        case .failure(let reason):
            message = reason
        case .success(let normalized):
            do {
                try engine.setServerUrl(url: normalized)
                reload()
                message = "Server URL saved."
                wakeDriver()
                checkConnection()
            } catch {
                message = "Could not save server URL: \(error.localizedDescription)"
            }
        }
    }

    /// Trim; empty clears the patient id; otherwise it must parse as a
    /// positive integer. Matches `app.rs:708-718` exactly, including leaving
    /// the stored value untouched when the text does not parse.
    func setPatientId(_ text: String) {
        guard let engine else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let parsed: Int64?
        if trimmed.isEmpty {
            parsed = nil
        } else if let value = Int64(trimmed), value > 0 {
            parsed = value
        } else {
            message = "Patient id must be a number — nothing will sync until it is."
            return
        }
        do {
            try engine.setPatientId(patientId: parsed)
            reload()
            message = nil
            wakeDriver()
        } catch {
            message = "Could not save patient id: \(error.localizedDescription)"
        }
    }

    func setSyncMode(_ mode: SyncMode) {
        guard let engine else { return }
        do {
            try engine.setSyncMode(mode: mode)
            reload()
            wakeDriver()
        } catch {
            message = "Could not save sync mode: \(error.localizedDescription)"
        }
    }

    /// Routed through `SyncController`, never through `KeychainTokenProvider`
    /// directly: the controller's `ForeignTokenStore` caches the token for
    /// the driver thread, and a direct Keychain write would leave that cache
    /// serving a stale token until relaunch. See `crates/apple/src/lib.rs`
    /// (`ForeignTokenStore`, `SyncController::set_token`).
    func saveToken(_ token: String) {
        guard let controller else { return }
        do {
            try controller.setToken(token: token.trimmingCharacters(in: .whitespacesAndNewlines))
            tokenStatus = "Token stored in the OS keychain."
            message = "Saved in the OS keychain."
        } catch {
            message = "Could not save token: \(error.localizedDescription)"
        }
    }

    /// "Sign in with PHR": runs the browser-based device-pairing flow
    /// (`PhrSignInSession`) instead of asking the user to paste a token by
    /// hand. Guards mirror `checkConnection`'s no-controller/already-running
    /// checks, plus one of its own — there is no server to pair with until a
    /// URL is saved.
    func signIn() {
        guard engine != nil, controller != nil, !isSigningIn else { return }
        if phr == nil {
            reload()
        }
        guard let phrSettings = phr else {
            message = "Could not read this device's settings — try again."
            return
        }
        guard !phrSettings.serverUrl.isEmpty else {
            message = "Set the server URL above before signing in."
            return
        }

        #if os(macOS)
        let deviceName = Host.current().localizedName ?? "Mac"
        #else
        let deviceName = UIDevice.current.name
        #endif

        isSigningIn = true
        message = nil
        let session = PhrSignInSession(
            serverRoot: phrSettings.serverUrl,
            deviceId: phrSettings.deviceId,
            deviceName: deviceName
        )
        // `session` is `@MainActor`-isolated, non-`Sendable` state; capturing
        // it into an explicitly `@MainActor` task closure is the same
        // established pattern as `start()`'s termination observer and
        // `TrainingModel`'s take-completion hops, not `Task.detached`'s
        // (`checkConnection`'s network call blocks a thread and so must
        // leave the main actor; this one only ever awaits, so it need not).
        Task { @MainActor [weak self] in
            let outcome = await session.signIn()
            self?.finishSignIn(outcome)
        }
    }

    private func finishSignIn(_ outcome: PhrSignInSession.Outcome) {
        isSigningIn = false
        switch outcome {
        case .success(let success):
            saveToken(success.token)
            message = "Signed in — this Mac now has its own key, expiring \(Self.formatExpiry(success.expiresAt))."
            checkConnection()
        case .failure(.cancelled):
            message = "Sign-in was cancelled."
        case .failure(.denied):
            message = "The pairing request was denied in the browser."
        case .failure(.invalidCode):
            message = "The pairing code was invalid or expired — try signing in again."
        case .failure(.network(let detail)):
            message = "Could not reach the server: \(detail)"
        case .failure(.server(let detail)):
            message = detail
        }
    }

    /// Best-effort human formatting of the exchange's ISO8601 `expires_at` —
    /// falls back to the raw string rather than hiding a token that did save
    /// just because its expiry did not parse.
    private static func formatExpiry(_ iso8601: String) -> String {
        guard let date = ISO8601DateFormatter().date(from: iso8601) else { return iso8601 }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    /// Checks only whether a token exists; never reads it into the UI —
    /// same promise the desktop tray's "Check token" hover text makes.
    ///
    /// This is deliberately not "is sync working": SPEC's motivating incident
    /// was a token that existed here and was still rejected by the server
    /// (revoked after a key-expiry policy landed on the PHR), and this check
    /// alone cannot see that — hence the wording pointing at `checkConnection`
    /// instead of implying validity it never had.
    func checkToken() {
        guard let controller else { return }
        do {
            tokenStatus = try controller.hasToken()
                ? "A token is stored in the OS keychain (existence only — use Test connection to validate it)."
                : "No API token is stored."
        } catch {
            tokenStatus = "Could not check token: \(error.localizedDescription)"
        }
    }

    /// Live "Test connection": asks the PHR whether the stored token is
    /// actually accepted for this patient, rather than only confirming one
    /// exists (see `checkToken`). Blocks on the network for up to the Rust
    /// HTTP client's 30-second timeout, so the call runs detached and hops
    /// back to the main actor only to publish the result.
    func checkConnection() {
        guard let controller, !isCheckingConnection else { return }
        isCheckingConnection = true
        // Mirrors `TrainingModel.finishTake`: call back into a `@MainActor`
        // method on `self?` rather than wrapping the publish in a nested
        // `MainActor.run` closure, which the Swift 6 concurrency checker
        // flags for capturing `self` across the actor hop.
        Task.detached { [weak self] in
            let outcome = controller.checkConnection()
            await self?.publishConnectionResult(outcome)
        }
    }

    private func publishConnectionResult(_ outcome: AppleConnectionCheck) {
        connectionStatus = ConnectionCheckMessage.message(for: outcome)
        connectionFailed = ConnectionCheckMessage.isFailure(outcome)
        isCheckingConnection = false
    }

    func removeToken() {
        guard let controller else { return }
        do {
            try controller.clearToken()
            tokenStatus = "No API token is stored."
            message = nil
        } catch {
            message = "Could not remove token: \(error.localizedDescription)"
        }
    }

    func syncNow() {
        controller?.syncNow()
    }

    /// What the tray app's `notify_sync` does after a connection edit: get the
    /// driver to notice, rather than leaving a corrected server URL unused
    /// until whatever tick it would otherwise have slept through.
    ///
    /// `sync_now` is the only wake the FFI exposes, and it is stronger — it
    /// also forces the flush. That is safe here because offline-strict is
    /// checked before the manual request in `should_flush`, so switching *to*
    /// offline-strict still cannot upload anything.
    private func wakeDriver() {
        controller?.syncNow()
    }

    var stateLabel: String {
        switch status.state {
        case .idle: return "idle"
        case .syncing: return "syncing…"
        case .failed: return "sync failing"
        }
    }

    var lastSuccessDescription: String {
        guard let ms = status.lastSuccessEpochMs else { return "never" }
        let date = Date(timeIntervalSince1970: Double(ms) / 1000)
        return Self.relativeFormatter.localizedString(for: date, relativeTo: Date())
    }

    /// The mapped text for `status.error`, or `nil` when there is none.
    var mappedError: String? {
        status.error.flatMap(humanReadableError)
    }

    /// The substring contract lives on `SyncStatusSnapshot.error` in
    /// `crates/apple/src/lib.rs` — do not turn this into structured
    /// error-code plumbing on the Rust side; the comment there says so
    /// explicitly. `"no API token configured"` means exactly that; this
    /// pane *is* Settings › PHR, so it points at the field above rather than
    /// telling the user to go somewhere they already are. `"keychain"`
    /// means the Keychain read itself failed. Anything else passes through.
    func humanReadableError(_ raw: String) -> String? {
        if raw.contains("no API token configured") {
            return "No API token is set — add one in the API token section above."
        }
        if raw.contains("keychain") {
            return "The Keychain read failed: \(raw)"
        }
        return raw
    }

    private static let relativeFormatter: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter
    }()
}

/// `SyncObserver` is called on the driver thread, not the main thread. A
/// `@MainActor` class cannot itself conform (the protocol is a plain
/// `Sendable`, not actor-isolated), so this non-isolated bridge exists only
/// to hop the callback onto the main actor. Cheap to do per-call: unlike the
/// 4 Hz status timer in `MonitorModel`, this fires once per sync tick.
private final class SyncStatusBridge: SyncObserver {
    private let handler: @Sendable (SyncStatusSnapshot) -> Void

    init(handler: @escaping @Sendable (SyncStatusSnapshot) -> Void) {
        self.handler = handler
    }

    func onStatus(status: SyncStatusSnapshot) {
        handler(status)
    }
}
