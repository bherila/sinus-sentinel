#if os(macOS)
import AppKit
import SwiftUI

enum WindowID {
    static let history = "history"
}

/// The `MenuBarExtra` body, pulled out of the `App` scene builder because it
/// needs `@Environment(\.openWindow)` / `@Environment(\.openSettings)`, neither
/// of which is available where `App.body` constructs its scenes.
struct MenuBarContent: View {
    @Environment(EngineHost.self) private var host
    @Environment(\.openWindow) private var openWindow
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        if host.monitor.blockedByOtherInstance {
            Text("Another Sinus Sentinel owns this computer")
        } else if host.monitor.suspendedForLowPower {
            Text("Paused for Low Power Mode")
        }
        Button(host.monitor.sessionRequested ? "Stop monitoring" : "Start monitoring") {
            host.monitor.toggleMonitoring()
        }
        .disabled(host.monitor.blockedByOtherInstance)
        Button("Open History") {
            AppDelegate.activateForWindow()
            openWindow(id: WindowID.history)
        }
        Button("Settings…") {
            AppDelegate.activateForWindow()
            openSettings()
        }
        .keyboardShortcut(",", modifiers: .command)
        Divider()
        Button("Quit") {
            NSApplication.shared.terminate(nil)
        }
    }
}
#endif
