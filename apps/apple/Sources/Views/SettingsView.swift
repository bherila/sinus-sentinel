import SwiftUI
import SinusAppleFFI

// The `Settings` scene and this fixed frame are both macOS-only concepts;
// iOS reaches the same panes through `RootTabView` / `IOSSettingsView`.
#if os(macOS)
struct SettingsView: View {
    var body: some View {
        TabView {
            GeneralSettingsView()
                .tabItem {
                    Label("General", systemImage: "gearshape")
                }
            PhrSettingsView()
                .tabItem {
                    Label("PHR", systemImage: "heart.text.square")
                }
            TrainingSettingsView()
                .tabItem {
                    Label("Training", systemImage: "waveform.badge.mic")
                }
            AboutSettingsView()
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
        }
        // Settings tabs share one frame, so this is sized for the taller PHR
        // pane (Connection + API token + Status sections), not General.
        .frame(width: 480, height: 520)
    }
}
#endif

struct GeneralSettingsView: View {
    @Environment(EngineHost.self) private var host

    @State private var quietHoursEnabled = false
    @State private var quietStartHour = 22
    @State private var quietEndHour = 7

    var body: some View {
        Form {
            Section("Detection") {
                Slider(
                    value: Binding(
                        get: { host.monitor.sensitivity },
                        set: { host.monitor.setSensitivity($0) }
                    ),
                    in: 0...1,
                    onEditingChanged: { editing in
                        // Only once the drag settles — syncing on every tick would
                        // push a document per frame. The value itself is already
                        // live on every change above, same split the desktop
                        // tray's slider makes between `changed()` and
                        // `drag_stopped()`.
                        if !editing {
                            host.sync.syncNow()
                        }
                    },
                    label: { Text("Sensitivity") }
                )
                .help("Shared with your other machines through the PHR")
            }

            Section("Quiet hours") {
                Toggle("Quiet hours", isOn: Binding(
                    get: { quietHoursEnabled },
                    set: { setQuietHoursEnabled($0) }
                ))
                if quietHoursEnabled {
                    Picker("Start hour (local)", selection: Binding(
                        get: { quietStartHour },
                        set: { quietStartHour = $0; writeQuietHours() }
                    )) {
                        ForEach(0..<24, id: \.self) { hour in
                            Text(Self.hourLabel(hour)).tag(hour)
                        }
                    }
                    Picker("End hour (local)", selection: Binding(
                        get: { quietEndHour },
                        set: { quietEndHour = $0; writeQuietHours() }
                    )) {
                        ForEach(0..<24, id: \.self) { hour in
                            Text(Self.hourLabel(hour)).tag(hour)
                        }
                    }
                    Text("Once the machine has been idle a few minutes inside this local-time window, the microphone is released. Using the machine resumes monitoring immediately, even inside the window. It may wrap past midnight — for example 22:00 to 07:00. Syncs with the PHR.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Off. Detections are recorded around the clock.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }

            Text("Sensitivity and quiet hours sync with the PHR, so they follow you between machines. Server URL, patient id and sync mode stay on this device.")
                .font(.footnote)
                .foregroundStyle(.secondary)

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
        .onAppear {
            if let hours = host.monitor.quietHours {
                quietHoursEnabled = true
                quietStartHour = Int(hours.startHour)
                quietEndHour = Int(hours.endHour)
            } else {
                quietHoursEnabled = false
            }
        }
    }

    private func setQuietHoursEnabled(_ enabled: Bool) {
        quietHoursEnabled = enabled
        if enabled {
            writeQuietHours()
        } else {
            host.monitor.setQuietHours(nil)
            host.sync.syncNow()
        }
    }

    private func writeQuietHours() {
        if quietStartHour == quietEndHour {
            // `start == end` is "no window" on the Rust side, not a one-hour
            // window — nudge the end forward rather than silently writing a
            // state the toggle just turned on to mean the opposite.
            quietEndHour = (quietEndHour + 1) % 24
        }
        host.monitor.setQuietHours(
            QuietHours(startHour: UInt32(quietStartHour), endHour: UInt32(quietEndHour))
        )
        host.sync.syncNow()
    }

    private static func hourLabel(_ hour: Int) -> String {
        String(format: "%02d:00", hour)
    }
}

struct AboutSettingsView: View {
    @Environment(EngineHost.self) private var host

    var body: some View {
        Form {
            Section("This machine") {
                LabeledContent("Device id") {
                    Text(host.sync.phr?.deviceId ?? "—")
                        .textSelection(.enabled)
                }
                .help("This machine's stable identity in the PHR — quote it in a support conversation")
                LabeledContent("Model version", value: host.monitor.status?.modelVersion ?? "—")
            }

            Section("Privacy") {
                Text("Audio is analyzed locally and is never stored or uploaded. Only events and embeddings sync to the PHR.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        // `SyncModel.phr` is only populated once `reload()` runs; if sync
        // failed to start (offline is a legitimate steady state — see
        // `EngineHost`), the initial load never happened even though the
        // engine itself is fine, so refresh here the same way
        // `PhrSettingsView` does.
        .onAppear { host.sync.reload() }
    }
}
