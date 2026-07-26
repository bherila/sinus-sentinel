import Foundation
import Observation
import SinusAppleFFI

/// Owns capture state and the battery policy that gates it — the power policy
/// *is* monitoring policy, so `LowPowerMonitor` lives here rather than beside it.
@MainActor
@Observable
final class MonitorModel {
    /// The user asked for a session. Capture may still be suspended — Low
    /// Power Mode, pause and quiet hours all suspend the microphone without
    /// discarding that intent, so the session resumes by itself once each
    /// condition lifts.
    private(set) var sessionRequested = false
    private(set) var isCapturing = false
    private(set) var suspendedForLowPower = false
    /// Distinct from `suspendedForLowPower` and from the user's `pause`: this
    /// reason ends when the machine notices activity again, not when Low
    /// Power Mode turns off or a pause deadline passes.
    private(set) var suppressedForQuietHours = false
    /// Another Sinus Sentinel owns this machine. The app stays open and inert
    /// rather than pretending to monitor alongside it.
    private(set) var blockedByOtherInstance = false
    private(set) var pauseOnLowPower = true
    private(set) var sensitivity: Float = 0.5
    private(set) var quietHours: QuietHours?
    /// The user's own pause, distinct from `suspendedForLowPower`: this one
    /// resumes on a deadline or an explicit Resume, not on Low Power Mode
    /// ending. Restored from the engine at `attach`, since a pause now
    /// survives relaunch.
    private(set) var pause: PauseSnapshot?
    /// Polled while capturing (for the "heard something — classifying…"
    /// indicator) and, more slowly, whenever a session is requested (to
    /// notice the quiet-hours window changing while the mic is released).
    /// `nil` before any read has happened.
    private(set) var status: EngineStatus?

    var isLowPowerModeEnabled: Bool { power.isLowPower }

    /// Raised whenever this model has written something the history projection
    /// would show: a live detection, or the tail a stopped session flushes. The
    /// host turns it into a refresh, so this model never learns that history
    /// exists.
    var onHistoryChanged: () -> Void = {}
    var onError: (String?) -> Void = { _ in }
    /// Raised when capture ends for any reason. A Teach take buffering at that
    /// moment can never fill, and it left event persistence suppressed on the
    /// Rust side — so somebody has to abandon it rather than let the detector
    /// quietly stop recording anything until relaunch.
    var onCaptureStopped: () -> Void = {}

    private var audio: AudioMonitoringService?
    private var engine: AppleEngine?
    private let power = LowPowerMonitor()
    /// Bound to capture, not to app lifetime: there is nothing to watch while
    /// the microphone is released, and a timer that kept running while
    /// suspended would defeat the battery policy this model exists to enforce.
    /// `@ObservationIgnored` also keeps it a plain stored property, which
    /// `deinit` needs to reach it outside actor isolation.
    @ObservationIgnored
    private var statusTimer: Timer?
    /// A timed pause ends by wall clock and nothing calls back on its own —
    /// `PauseSnapshot.paused` is only re-derived when read. This timer is what
    /// notices the deadline instead of a poll loop that would not even be
    /// running once the microphone is released.
    @ObservationIgnored
    private var pauseExpiryTimer: Timer?
    /// Runs whenever a session is requested, independent of whether capture is
    /// currently live — the opposite lifetime rule from `statusTimer` above.
    /// Quiet hours (like Low Power Mode and pause) can release the microphone
    /// for hours; without a poll that keeps running through that release,
    /// nothing would ever notice the user coming back, and monitoring would
    /// stay off until they toggled it by hand. 30s matches the desktop
    /// capture worker's `QUIET_POLL_INTERVAL` (`apps/desktop/src/capture.rs`)
    /// and its shortened `suspension_wait` park. `AppleEngine.status()` never
    /// takes the writer lock, so polling it while suspended is cheap and safe.
    @ObservationIgnored
    private var quietHoursPollTimer: Timer?

    /// How long the machine must go untouched before quiet hours treat the
    /// user as absent. Mirrors `sinus_app::sync::IDLE_FOR_QUIET_HOURS`
    /// (crates/app/src/sync.rs) — that constant is the authority; this is a
    /// copy because the threshold is not exposed over the FFI.
    private static let idleForQuietHours: TimeInterval = 5 * 60

    /// Mirrors `sinus_app::sync::suppress_for_quiet_hours` (crates/app/src/sync.rs):
    /// the window is a proxy for absence, not an instruction to ignore the
    /// clock, so someone demonstrably at the keyboard inside their own quiet
    /// window is still monitored. `idle == nil` (no honest signal) falls back
    /// to the literal window.
    private static func suppressForQuietHours(inWindow: Bool, idle: TimeInterval?) -> Bool {
        guard inWindow else { return false }
        guard let idle else { return true }
        return idle >= idleForQuietHours
    }

    init() {
        power.onChange = { [weak self] _ in
            self?.applyPolicy()
        }
    }

    deinit {
        statusTimer?.invalidate()
        pauseExpiryTimer?.invalidate()
        quietHoursPollTimer?.invalidate()
    }

    func attach(engine: AppleEngine, audio: AudioMonitoringService) {
        self.engine = engine
        self.audio = audio
        audio.onEvents = { [weak self] _ in
            Task { @MainActor in
                self?.onHistoryChanged()
            }
        }
        audio.onFailure = { [weak self] message in
            Task { @MainActor in
                self?.handleCaptureFailure(message)
            }
        }
        pauseOnLowPower = (try? engine.pauseOnLowPower()) ?? true
        sensitivity = (try? engine.sensitivity()) ?? sensitivity
        quietHours = try? engine.quietHours()
        pause = try? engine.pause()
        schedulePauseExpiry()
        pollStatus()
    }

    func markBlockedByOtherInstance() {
        blockedByOtherInstance = true
    }

    /// Not a `didSet`: `@Observable` cannot transform a property that already has
    /// an accessor, so the write and its side effects are spelled out instead.
    func setPauseOnLowPower(_ enabled: Bool) {
        guard enabled != pauseOnLowPower else { return }
        pauseOnLowPower = enabled
        try? engine?.setPauseOnLowPower(enabled: enabled)
        applyPolicy()
    }

    /// Written live so the running detector picks it up immediately
    /// (`set_sensitivity` reloads it); syncing to the PHR is the caller's
    /// job, on drag-end, the same split the desktop tray's slider makes.
    func setSensitivity(_ value: Float) {
        sensitivity = value
        try? engine?.setSensitivity(sensitivity: value)
    }

    /// `nil` clears the window. `start == end` is "no window" on the Rust
    /// side, not a one-hour window — callers must not pass that.
    func setQuietHours(_ hours: QuietHours?) {
        guard hours != quietHours else { return }
        quietHours = hours
        try? engine?.setQuietHours(hours: hours)
    }

    /// `set_pause` only persists the state; it does not touch capture. Pause
    /// is a third input to `applyPolicy`, alongside session intent and Low
    /// Power Mode, for the same reason that policy releases the microphone
    /// rather than merely skipping analysis: the tray app's own pause
    /// enforcement (`capture.rs:196-209`) does the same.
    func setPause(kind: PauseKind, until: Date?) {
        guard let engine else { return }
        let untilMs = until.map { Int64(($0.timeIntervalSince1970 * 1000).rounded()) }
        do {
            try engine.setPause(kind: kind, untilEpochMs: untilMs)
            pause = try? engine.pause()
            schedulePauseExpiry()
            applyPolicy()
        } catch {
            onError(error.localizedDescription)
        }
    }

    func toggleMonitoring() {
        if sessionRequested {
            sessionRequested = false
            suspendedForLowPower = false
            suppressedForQuietHours = false
            stopQuietHoursPolling()
            stopCapture()
            return
        }
        Task {
            guard await AudioMonitoringService.requestPermission() else {
                onError("Microphone access is required for a monitoring session.")
                return
            }
            sessionRequested = true
            onError(nil)
            startQuietHoursPolling()
            // Before the first policy decision, not after: `attach` read the
            // status at launch, and whether the local hour is inside the quiet
            // window will have gone stale by the time anyone presses start.
            pollStatus()
            applyPolicy()
        }
    }

    /// The single place that decides whether the microphone should be live:
    /// "the user wants a session", "the OS wants us to use less power", "the
    /// user paused", and "quiet hours judge the user away". Called on every
    /// input that can change any of them.
    private func applyPolicy() {
        guard sessionRequested else { return }
        let shouldSuspendForPower = pauseOnLowPower && power.isLowPower
        suspendedForLowPower = shouldSuspendForPower
        let userPaused = pause?.paused ?? false
        let shouldSuppressForQuietHours = Self.suppressForQuietHours(
            inWindow: status?.inQuietHours ?? false,
            idle: UserIdleMonitor.idle()
        )
        suppressedForQuietHours = shouldSuppressForQuietHours
        if shouldSuspendForPower || userPaused || shouldSuppressForQuietHours {
            // Releasing the microphone — not merely skipping analysis — is what
            // actually lets the audio hardware and its wake-ups idle.
            stopCapture()
        } else if !isCapturing {
            startCapture()
        }
    }

    private func startCapture() {
        guard let audio else { return }
        do {
            try audio.start()
            isCapturing = true
            // Seed status before the first status-timer tick so the
            // "heard something — classifying…" indicator has fresh state from
            // the first frame rather than a stale read from before capture
            // began.
            pollStatus()
            startStatusPolling()
            onError(nil)
        } catch {
            sessionRequested = false
            isCapturing = false
            stopQuietHoursPolling()
            onError(error.localizedDescription)
        }
    }

    private func stopCapture() {
        guard let audio, isCapturing else { return }
        stopStatusPolling()
        onCaptureStopped()
        do {
            _ = try audio.stop()
            isCapturing = false
            // One last read after the session closes. Without it the UI keeps
            // whatever the final tick saw, and a gate that happened to be open
            // when the user hit stop would sit there claiming to be classifying
            // a sound from a microphone that is no longer running.
            pollStatus()
            onHistoryChanged()
        } catch {
            isCapturing = false
            onError(error.localizedDescription)
        }
    }

    private func startStatusPolling() {
        stopStatusPolling()
        statusTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            // Timer callbacks run on the main run loop already; a `Task { @MainActor
            // in }` hop four times a second would be pure overhead for isolation
            // that already holds.
            MainActor.assumeIsolated {
                self?.pollStatus()
            }
        }
    }

    private func stopStatusPolling() {
        statusTimer?.invalidate()
        statusTimer = nil
    }

    /// Unlike `startStatusPolling`, this timer's lifetime is tied to
    /// `sessionRequested`, not to `isCapturing` — see the property comment on
    /// `quietHoursPollTimer` for why.
    private func startQuietHoursPolling() {
        stopQuietHoursPolling()
        quietHoursPollTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.pollStatus()
                self?.applyPolicy()
            }
        }
    }

    private func stopQuietHoursPolling() {
        quietHoursPollTimer?.invalidate()
        quietHoursPollTimer = nil
    }

    private func pollStatus() {
        guard let engine else { return }
        status = try? engine.status()
    }

    private func schedulePauseExpiry() {
        pauseExpiryTimer?.invalidate()
        pauseExpiryTimer = nil
        guard let pause, pause.kind == .timed, pause.paused, let untilMs = pause.untilEpochMs else { return }
        let deadline = Date(timeIntervalSince1970: Double(untilMs) / 1000)
        pauseExpiryTimer = Timer.scheduledTimer(
            withTimeInterval: max(deadline.timeIntervalSinceNow, 0.01),
            repeats: false
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.handlePauseExpired()
            }
        }
    }

    private func handlePauseExpired() {
        guard let engine else { return }
        pause = try? engine.pause()
        applyPolicy()
    }

    private func handleCaptureFailure(_ message: String) {
        sessionRequested = false
        suspendedForLowPower = false
        suppressedForQuietHours = false
        stopQuietHoursPolling()
        stopCapture()
        onError(message)
    }
}
