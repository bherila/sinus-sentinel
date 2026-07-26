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

                todaySection

                VStack(alignment: .leading, spacing: 8) {
                    Text("Last 7 days")
                        .font(.title2.bold())
                    HistoryChartView(snapshot: host.history.snapshot)
                }

                RecentEventsView()
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

    @ViewBuilder
    private var todaySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Today")
                .font(.title2.bold())

            let counts = (host.history.snapshot?.today ?? []).filter { $0.count > 0 }
            if counts.isEmpty {
                Text("no events yet today")
                    .foregroundStyle(.secondary)
            } else {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 120), alignment: .leading)],
                    alignment: .leading,
                    spacing: 6
                ) {
                    ForEach(counts, id: \.eventType) { count in
                        HStack(spacing: 4) {
                            Rectangle()
                                .fill(count.eventType.color)
                                .frame(width: 10, height: 10)
                            Text("\(count.eventType.displayName): \(count.count)")
                        }
                    }
                }
            }

            if let snapshot = host.history.snapshot {
                Text(
                    "Congestion score: \(snapshot.congestionScorePerMonitoredHour, specifier: "%.2f") per monitored hour (\(snapshot.monitoredHours, specifier: "%.1f") monitored hours)"
                )
                .font(.subheadline)
                .foregroundStyle(.secondary)
            }
        }
    }
}
