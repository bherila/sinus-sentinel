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
        MenuBarExtra("Sinus Sentinel", systemImage: model.isMonitoring ? "waveform" : "pause.circle") {
            Button(model.isMonitoring ? "Stop monitoring" : "Start monitoring") {
                model.toggleMonitoring()
            }
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
