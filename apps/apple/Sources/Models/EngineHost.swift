import Foundation
import Observation
import SinusAppleFFI

/// Owns everything that must exist exactly once per process: the `AppleEngine`
/// (and with it this machine's instance lock), the audio capture service, and
/// the models built on top of them.
@MainActor
@Observable
final class EngineHost {
    let monitor: MonitorModel
    let history: HistoryModel
    let sync: SyncModel
    var errorMessage: String?

    init() {
        let monitor = MonitorModel()
        let history = HistoryModel()
        let sync = SyncModel()
        self.monitor = monitor
        self.history = history
        self.sync = sync
        monitor.onError = { [weak self] in self?.errorMessage = $0 }
        history.onError = { [weak self] in self?.errorMessage = $0 }
        // Closes chunk 12's gap: the tray app requests a sync after every
        // flag, and Swift did not. A hook rather than a reference to
        // `SyncModel`, so `HistoryModel` keeps not knowing what else exists.
        history.onFlagged = { [weak self] in self?.sync.syncNow() }

        do {
            let support = try Self.applicationSupportDirectory()
            let database = support.appendingPathComponent("events.db")
            let runner = Self.modelRunner()
            let engine = try AppleEngine(
                config: AppleEngineConfig(
                    databasePath: database.path,
                    platform: Self.platform
                ),
                model: runner
            )
            let audio = AudioMonitoringService(engine: engine)
            history.attach(engine: engine)
            monitor.attach(engine: engine, audio: audio)
            monitor.onHistoryChanged = { [weak history] in history?.refresh() }
            history.refresh()

            do {
                try sync.start(engine: engine, tokens: KeychainTokenProvider())
            } catch {
                // Offline is a legitimate steady state: monitoring and
                // history keep working with no PHR connection, and every
                // `SyncModel` write below no-ops without a controller — the
                // same shape `MonitorModel`/`HistoryModel` use for a nil
                // engine.
                errorMessage = error.localizedDescription
            }
        } catch AppleEngineError.AlreadyRunning {
            monitor.markBlockedByOtherInstance()
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

    private static func modelRunner() -> ModelRunner {
        if let compiled = Bundle.main.url(forResource: "yamnet", withExtension: "mlmodelc"),
           let runner = try? CoreMLYamnetRunner(compiledModelURL: compiled) {
            return runner
        }
        return PreviewModelRunner()
    }
}
