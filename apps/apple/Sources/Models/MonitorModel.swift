import Foundation
import Observation
import SinusAppleFFI

/// Owns capture state and the battery policy that gates it — the power policy
/// *is* monitoring policy, so `LowPowerMonitor` lives here rather than beside it.
@MainActor
@Observable
final class MonitorModel {
    /// The user asked for a session. Capture may still be suspended — battery
    /// policy suspends the microphone without discarding that intent, so the
    /// session resumes by itself when the device leaves Low Power Mode.
    private(set) var sessionRequested = false
    private(set) var isCapturing = false
    private(set) var suspendedForLowPower = false
    /// Another Sinus Sentinel owns this machine. The app stays open and inert
    /// rather than pretending to monitor alongside it.
    private(set) var blockedByOtherInstance = false
    private(set) var pauseOnLowPower = true
    /// Polled while capturing, for the "heard something — classifying…"
    /// indicator. `nil` before any read has happened.
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

    init() {
        power.onChange = { [weak self] _ in
            self?.applyPowerPolicy()
        }
    }

    deinit {
        statusTimer?.invalidate()
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
        applyPowerPolicy()
    }

    func toggleMonitoring() {
        if sessionRequested {
            sessionRequested = false
            suspendedForLowPower = false
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
            applyPowerPolicy()
        }
    }

    /// The single place that reconciles "the user wants a session" with "the OS
    /// wants us to use less power". Called on every input that can change either.
    private func applyPowerPolicy() {
        guard sessionRequested else { return }
        let shouldSuspend = pauseOnLowPower && power.isLowPower
        suspendedForLowPower = shouldSuspend
        if shouldSuspend {
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
            startStatusPolling()
            onError(nil)
        } catch {
            sessionRequested = false
            isCapturing = false
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

    private func pollStatus() {
        guard let engine else { return }
        status = try? engine.status()
    }

    private func handleCaptureFailure(_ message: String) {
        sessionRequested = false
        suspendedForLowPower = false
        stopCapture()
        onError(message)
    }
}
