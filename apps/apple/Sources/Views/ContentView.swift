import SwiftUI

struct ContentView: View {
    @Environment(EngineHost.self) private var host

    var body: some View {
        NavigationStack {
            if host.monitor.blockedByOtherInstance {
                alreadyRunning
            } else {
                main
            }
        }
        .frame(minWidth: 480, minHeight: 540)
    }

    private var alreadyRunning: some View {
        ContentUnavailableView {
            Label("Sinus Sentinel is already running", systemImage: "exclamationmark.triangle")
        } description: {
            Text(
                "Another copy of Sinus Sentinel owns this computer's history. Quit it from the menu bar, then reopen this app — two apps listening at once would record every cough twice."
            )
        }
        .navigationTitle("Sinus Sentinel")
    }

    private var main: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                monitoringCard

                VStack(alignment: .leading, spacing: 8) {
                    Text("Last 7 days")
                        .font(.title2.bold())
                    HistoryChartView(snapshot: host.history.snapshot)
                }

                if let snapshot = host.history.snapshot {
                    Text(
                        "Congestion score: \(snapshot.congestionScorePerMonitoredHour, specifier: "%.2f") per monitored hour"
                    )
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                }

                recentEvents
                batterySettings
            }
            .padding()
        }
        .navigationTitle("Sinus Sentinel")
        .alert(
            "Sinus Sentinel",
            isPresented: Binding(
                get: { host.errorMessage != nil },
                set: { if !$0 { host.errorMessage = nil } }
            )
        ) {
            Button("OK") { host.errorMessage = nil }
        } message: {
            Text(host.errorMessage ?? "")
        }
    }

    private var monitoringCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(status.title, systemImage: status.symbol)
                .font(.title2.bold())
                .foregroundStyle(status.tint)

            Text(status.detail)
                .foregroundStyle(.secondary)

            Button(host.monitor.sessionRequested ? "Stop monitoring" : "Start monitoring") {
                host.monitor.toggleMonitoring()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var status: (title: String, detail: String, symbol: String, tint: Color) {
        if host.monitor.suspendedForLowPower {
            return (
                "Paused for Low Power Mode",
                "The microphone is released while the device saves power. Monitoring resumes on its own when Low Power Mode turns off.",
                "battery.25",
                .orange
            )
        }
        if host.monitor.isCapturing {
            return (
                "Monitoring is active",
                "The session continues when the iPhone locks. Audio is analyzed locally and is never stored.",
                "waveform.circle.fill",
                .green
            )
        }
        return (
            "Ready to monitor",
            "Start an explicit session when you want Sinus Sentinel to listen.",
            "waveform.circle",
            .primary
        )
    }

    private var batterySettings: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Battery")
                .font(.title2.bold())
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

    @ViewBuilder
    private var recentEvents: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent")
                .font(.title2.bold())
            let events = host.history.snapshot?.recentEvents ?? []
            if events.isEmpty {
                Text("No recent events")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(events.prefix(10), id: \.uuid) { event in
                    HStack {
                        Text(event.eventType.displayName)
                        Spacer()
                        Text(
                            Date(timeIntervalSince1970: Double(event.occurredAtEpochMs) / 1_000),
                            style: .time
                        )
                        .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }
}
