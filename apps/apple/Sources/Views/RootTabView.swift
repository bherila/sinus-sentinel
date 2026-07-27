#if os(iOS)
import SwiftUI

/// The iPhone shell around the shared views: a `TabView` standing in for the
/// separate history window / Settings scene / menu bar that macOS gets from
/// `SinusSentinelApp`. Each tab owns its own `NavigationStack` — except
/// History, which reuses `ContentView`'s own stack rather than nesting a
/// second one.
struct RootTabView: View {
    var body: some View {
        TabView {
            ContentView()
                .tabItem {
                    Label("History", systemImage: "chart.bar")
                }

            NavigationStack {
                TrainingSettingsView()
                    .navigationTitle("Train")
            }
            .tabItem {
                Label("Train", systemImage: "waveform.badge.mic")
            }

            NavigationStack {
                IOSSettingsView()
            }
            .tabItem {
                Label("Settings", systemImage: "gearshape")
            }
        }
    }
}

/// Settings › root, iOS. `GeneralSettingsView`, `PhrSettingsView` and
/// `AboutSettingsView` are each already a `Form`, and nesting Forms renders
/// badly — so this is a plain list of rows pushing to each one, standing in
/// for the macOS `Settings` scene's `TabView`.
struct IOSSettingsView: View {
    var body: some View {
        List {
            NavigationLink("General") {
                GeneralSettingsView()
                    .navigationTitle("General")
            }
            NavigationLink("PHR") {
                PhrSettingsView()
                    .navigationTitle("PHR")
            }
            NavigationLink("About") {
                AboutSettingsView()
                    .navigationTitle("About")
            }
        }
        .navigationTitle("Settings")
    }
}
#endif
