#if os(macOS)
import AppKit
import SwiftUI
import SinusAppleFFI

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
        } else if host.monitor.pause?.paused == true {
            // Distinct from Low Power suspension below: this one resumes on a
            // deadline or an explicit Resume, not on Low Power Mode ending.
            Text(pauseStatusLine)
        } else if host.monitor.suspendedForLowPower {
            Text("Paused for Low Power Mode")
        } else if host.monitor.suppressedForQuietHours {
            // Distinct from the two cases above: this one ends the moment the
            // machine notices activity again, not on a deadline or a mode change.
            Text("Quiet hours — away")
        }
        Button(host.monitor.sessionRequested ? "Stop monitoring" : "Start monitoring") {
            host.monitor.toggleMonitoring()
        }
        .disabled(host.monitor.blockedByOtherInstance)
        Menu("Pause") {
            Button("Pause 15 min") {
                host.monitor.setPause(kind: .timed, until: Date().addingTimeInterval(15 * 60))
            }
            Button("Pause 1 hour") {
                host.monitor.setPause(kind: .timed, until: Date().addingTimeInterval(60 * 60))
            }
            Button("Pause until resumed") {
                host.monitor.setPause(kind: .indefinite, until: nil)
            }
            if host.monitor.pause?.paused == true {
                Button("Resume") {
                    host.monitor.setPause(kind: .running, until: nil)
                }
            }
        }
        // A `Picker` placed directly in menu content renders as a native
        // submenu with the current selection checkmarked — no extra `Menu`
        // wrapper needed.
        Picker("Mode", selection: Binding(
            get: { host.sync.phr?.mode ?? .autoBatch },
            set: { host.sync.setSyncMode($0) }
        )) {
            Text("Auto-batch").tag(SyncMode.autoBatch)
            Text("Offline-first").tag(SyncMode.offlineFirst)
            Text("Offline-strict").tag(SyncMode.offlineStrict)
        }
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
        // A bare `Text`, like the status lines above — not a `Button` — so it
        // renders as a disabled menu entry, matching the desktop tray's
        // non-interactive "sync: …" toolbar label (`app.rs:1126-1130`).
        Text("sync: \(host.sync.stateLabel) (\(host.sync.status.pendingEvents) pending)")
        Button("Sync now") {
            host.sync.syncNow()
        }
        .help("flush pending events to the PHR now")
        Divider()
        Button("Quit") {
            NSApplication.shared.terminate(nil)
        }
    }

    private var pauseStatusLine: String {
        guard let pause = host.monitor.pause, pause.paused else { return "" }
        guard pause.kind == .timed, let untilMs = pause.untilEpochMs else {
            return "Paused until resumed"
        }
        let until = Date(timeIntervalSince1970: Double(untilMs) / 1000)
        return "Paused until \(until.formatted(date: .omitted, time: .shortened))"
    }
}
#endif
