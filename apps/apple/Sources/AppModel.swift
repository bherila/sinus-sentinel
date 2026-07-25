import AVFoundation
import Foundation
import SinusAppleFFI
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    /// The user asked for a session. Capture may still be suspended — battery
    /// policy suspends the microphone without discarding that intent, so the
    /// session resumes by itself when the device leaves Low Power Mode.
    @Published private(set) var sessionRequested = false
    @Published private(set) var isCapturing = false
    @Published private(set) var suspendedForLowPower = false
    @Published private(set) var snapshot: HistorySnapshot?
    @Published var errorMessage: String?

    @Published var pauseOnLowPower = true {
        didSet {
            guard pauseOnLowPower != oldValue else { return }
            try? engine?.setPauseOnLowPower(enabled: pauseOnLowPower)
            applyPowerPolicy()
        }
    }

    var isLowPowerModeEnabled: Bool { power.isLowPower }

    private var audio: AudioMonitoringService?
    private var engine: AppleEngine?
    private let power = LowPowerMonitor()

    init() {
        do {
            let support = try Self.applicationSupportDirectory()
            let database = support.appendingPathComponent("events.db")
            let runner = Self.modelRunner()
            let engine = try AppleEngine(
                config: AppleEngineConfig(
                    databasePath: database.path,
                    deviceId: Self.deviceID(),
                    platform: Self.platform,
                    sensitivity: 0.5
                ),
                model: runner
            )
            let audio = AudioMonitoringService(engine: engine)
            audio.onEvents = { [weak self] _ in
                Task { @MainActor in
                    self?.refreshHistory()
                }
            }
            audio.onFailure = { [weak self] message in
                Task { @MainActor in
                    self?.handleCaptureFailure(message)
                }
            }
            self.engine = engine
            self.audio = audio
            pauseOnLowPower = (try? engine.pauseOnLowPower()) ?? true
            power.onChange = { [weak self] _ in
                self?.applyPowerPolicy()
            }
            refreshHistory()
        } catch {
            errorMessage = error.localizedDescription
        }
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
                errorMessage = "Microphone access is required for a monitoring session."
                return
            }
            sessionRequested = true
            errorMessage = nil
            applyPowerPolicy()
        }
    }

    func refreshHistory() {
        guard let engine else { return }
        do {
            snapshot = try engine.history(
                days: 7,
                nowEpochMs: Self.nowMilliseconds,
                timezoneOffsetMinutes: Int32(TimeZone.current.secondsFromGMT() / 60)
            )
        } catch {
            errorMessage = error.localizedDescription
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
            errorMessage = nil
        } catch {
            sessionRequested = false
            isCapturing = false
            errorMessage = error.localizedDescription
        }
    }

    private func stopCapture() {
        guard let audio, isCapturing else { return }
        do {
            _ = try audio.stop()
            isCapturing = false
            refreshHistory()
        } catch {
            isCapturing = false
            errorMessage = error.localizedDescription
        }
    }

    private func handleCaptureFailure(_ message: String) {
        sessionRequested = false
        suspendedForLowPower = false
        stopCapture()
        errorMessage = message
    }

    private static var platform: ApplePlatform {
        #if os(iOS)
        return .ios
        #else
        return .macos
        #endif
    }

    private static var nowMilliseconds: Int64 {
        Int64(Date().timeIntervalSince1970 * 1_000)
    }

    private static func applicationSupportDirectory() throws -> URL {
        let root = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
        let directory = root.appendingPathComponent("SinusSentinel", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }

    private static func deviceID() -> String {
        let key = "sinus-sentinel-device-id"
        if let existing = UserDefaults.standard.string(forKey: key) {
            return existing
        }
        let created = UUID().uuidString
        UserDefaults.standard.set(created, forKey: key)
        return created
    }

    private static func modelRunner() -> ModelRunner {
        if let compiled = Bundle.main.url(forResource: "yamnet", withExtension: "mlmodelc"),
           let runner = try? CoreMLYamnetRunner(compiledModelURL: compiled) {
            return runner
        }
        return PreviewModelRunner()
    }
}
