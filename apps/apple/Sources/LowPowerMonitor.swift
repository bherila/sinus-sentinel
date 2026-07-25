import Foundation

/// Watches the OS battery-saver switch — Low Power Mode on iOS, Low Power Mode
/// on Apple silicon Macs — and reports transitions on the main actor.
///
/// The same signal drives both platforms: `ProcessInfo.isLowPowerModeEnabled` is
/// available on iOS 9+ and macOS 12+, and the system posts
/// `NSProcessInfoPowerStateDidChange` whenever the user (or a low battery) flips
/// it. Nothing here polls, so an idle app costs nothing.
@MainActor
final class LowPowerMonitor {
    /// Called on every transition, and never for a no-op change.
    var onChange: ((Bool) -> Void)?

    private(set) var isLowPower: Bool
    private var observer: NSObjectProtocol?

    init() {
        isLowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        observer = NotificationCenter.default.addObserver(
            forName: .NSProcessInfoPowerStateDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.powerStateChanged()
            }
        }
    }

    deinit {
        if let observer {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    private func powerStateChanged() {
        let current = ProcessInfo.processInfo.isLowPowerModeEnabled
        guard current != isLowPower else { return }
        isLowPower = current
        onChange?(current)
    }
}
