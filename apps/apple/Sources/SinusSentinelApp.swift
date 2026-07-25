import SwiftUI

#if os(macOS)
import AppKit
#endif

@main
struct SinusSentinelApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
        }

        #if os(macOS)
        MenuBarExtra("Sinus Sentinel", systemImage: model.isCapturing ? "waveform" : "pause.circle") {
            if model.blockedByOtherInstance {
                Text("Another Sinus Sentinel owns this computer")
            } else if model.suspendedForLowPower {
                Text("Paused for Low Power Mode")
            }
            Button(model.sessionRequested ? "Stop monitoring" : "Start monitoring") {
                model.toggleMonitoring()
            }
            .disabled(model.blockedByOtherInstance)
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
