import SwiftUI

#if os(macOS)
import AppKit
#endif

@main
struct SinusSentinelApp: App {
    @State private var host = EngineHost()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(host)
        }

        #if os(macOS)
        MenuBarExtra("Sinus Sentinel", systemImage: host.monitor.isCapturing ? "waveform" : "pause.circle") {
            if host.monitor.blockedByOtherInstance {
                Text("Another Sinus Sentinel owns this computer")
            } else if host.monitor.suspendedForLowPower {
                Text("Paused for Low Power Mode")
            }
            Button(host.monitor.sessionRequested ? "Stop monitoring" : "Start monitoring") {
                host.monitor.toggleMonitoring()
            }
            .disabled(host.monitor.blockedByOtherInstance)
            Button("Open Sinus Sentinel") {
                NSApplication.shared.activate(ignoringOtherApps: true)
            }
            Divider()
            Button("Quit") {
                NSApplication.shared.terminate(nil)
            }
        }
        #endif
    }
}
