import Foundation
import Observation
import SinusAppleFFI

#if os(macOS)
import AppKit
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

    private var engine: AppleEngine?
    private var controller: SyncController?

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
        let observer = SyncStatusBridge { [weak self] status in
            Task { @MainActor in
                self?.status = status
            }
        }
        let controller = try SyncController(engine: engine, tokens: tokens, observer: observer)
        self.controller = controller
        status = controller.status()
        reload()

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

    func shutdown() {
        controller?.shutdown(timeoutMs: 3000)
    }

    func reload() {
        guard let engine else { return }
        phr = try? engine.phrSettings()
    }

    func setServerUrl(_ url: String) {
        guard let engine else { return }
        do {
            try engine.setServerUrl(url: url)
            reload()
            wakeDriver()
        } catch {
            message = "Could not save server URL: \(error.localizedDescription)"
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

    /// Checks only whether a token exists; never reads it into the UI —
    /// same promise the desktop tray's "Check token" hover text makes.
    func checkToken() {
        guard let controller else { return }
        do {
            tokenStatus = try controller.hasToken()
                ? "Token stored in the OS keychain."
                : "No API token is stored."
        } catch {
            tokenStatus = "Could not check token: \(error.localizedDescription)"
        }
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
