import AVFoundation
import Foundation
import SinusAppleFFI
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var isMonitoring = false
    @Published private(set) var snapshot: HistorySnapshot?
    @Published private(set) var latestEvents: [AppleEvent] = []
    @Published var errorMessage: String?

    private var audio: AudioMonitoringService?
    private var engine: AppleEngine?

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
            audio.onEvents = { [weak self] events in
                Task { @MainActor in
                    self?.latestEvents = events + (self?.latestEvents ?? [])
                    self?.refreshHistory()
                }
            }
            audio.onFailure = { [weak self] message in
                Task { @MainActor in
                    self?.isMonitoring = false
                    self?.errorMessage = message
                }
            }
            self.engine = engine
            self.audio = audio
            refreshHistory()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func toggleMonitoring() {
        if isMonitoring {
            stopMonitoring()
        } else {
            Task {
                guard await AudioMonitoringService.requestPermission() else {
                    errorMessage = "Microphone access is required for a monitoring session."
                    return
                }
                startMonitoring()
            }
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

    private func startMonitoring() {
        guard let audio else { return }
        do {
            try audio.start()
            isMonitoring = true
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func stopMonitoring() {
        guard let audio else { return }
        do {
            let events = try audio.stop()
            latestEvents = events + latestEvents
            isMonitoring = false
            refreshHistory()
        } catch {
            errorMessage = error.localizedDescription
        }
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
