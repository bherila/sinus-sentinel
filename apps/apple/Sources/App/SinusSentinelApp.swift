import SwiftUI

@main
struct SinusSentinelApp: App {
    @State private var host = EngineHost()

    #if os(macOS)
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    #else
    @Environment(\.scenePhase) private var scenePhase
    #endif

    var body: some Scene {
        #if os(macOS)
        // `Window`, not `WindowGroup`: there is one history per machine, and a
        // group would let the user open several views of it that then have to
        // be kept in agreement.
        Window("Sinus Sentinel", id: WindowID.history) {
            ContentView()
                .environment(host)
        }

        Settings {
            SettingsView()
                .environment(host)
        }

        MenuBarExtra("Sinus Sentinel", systemImage: host.monitor.isCapturing ? "waveform" : "pause.circle") {
            MenuBarContent()
                .environment(host)
        }
        #else
        WindowGroup {
            RootTabView()
                .environment(host)
                .onChange(of: scenePhase) { _, newPhase in
                    // `.inactive` is the transient state on either side of a
                    // phone call or the app switcher — neither backgrounding
                    // nor foregrounding actually happened, so it is ignored.
                    switch newPhase {
                    case .background:
                        host.enterBackground()
                    case .active:
                        host.enterForeground()
                    case .inactive:
                        break
                    @unknown default:
                        break
                    }
                }
        }
        #endif
    }
}
