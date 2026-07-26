import SwiftUI

struct SettingsView: View {
    var body: some View {
        TabView {
            GeneralSettingsView()
                .tabItem {
                    Label("General", systemImage: "gearshape")
                }
        }
        .frame(width: 420, height: 260)
    }
}

private struct GeneralSettingsView: View {
    @Environment(EngineHost.self) private var host

    var body: some View {
        Form {
            Section("Battery") {
                Toggle("Pause while Low Power Mode is on", isOn: Binding(
                    get: { host.monitor.pauseOnLowPower },
                    set: { host.monitor.setPauseOnLowPower($0) }
                ))
                Text(
                    host.monitor.isLowPowerModeEnabled
                        ? "Low Power Mode is on right now."
                        : "Releases the microphone and resumes automatically. Shared with the desktop app through the same database."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}
