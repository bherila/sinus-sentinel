import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    monitoringCard

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Last 7 days")
                            .font(.title2.bold())
                        HistoryChartView(snapshot: model.snapshot)
                    }

                    if let snapshot = model.snapshot {
                        Text(
                            "Congestion score: \(snapshot.congestionScorePerMonitoredHour, specifier: "%.2f") per monitored hour"
                        )
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    }

                    recentEvents
                }
                .padding()
            }
            .navigationTitle("Sinus Sentinel")
            .alert(
                "Sinus Sentinel",
                isPresented: Binding(
                    get: { model.errorMessage != nil },
                    set: { if !$0 { model.errorMessage = nil } }
                )
            ) {
                Button("OK") { model.errorMessage = nil }
            } message: {
                Text(model.errorMessage ?? "")
            }
        }
        .frame(minWidth: 480, minHeight: 540)
    }

    private var monitoringCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(
                model.isMonitoring ? "Monitoring is active" : "Ready to monitor",
                systemImage: model.isMonitoring ? "waveform.circle.fill" : "waveform.circle"
            )
            .font(.title2.bold())
            .foregroundStyle(model.isMonitoring ? .green : .primary)

            Text(
                model.isMonitoring
                    ? "The session continues when the iPhone locks. Audio is analyzed locally and is never stored."
                    : "Start an explicit session when you want Sinus Sentinel to listen."
            )
            .foregroundStyle(.secondary)

            Button(model.isMonitoring ? "Stop monitoring" : "Start monitoring") {
                model.toggleMonitoring()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    @ViewBuilder
    private var recentEvents: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent")
                .font(.title2.bold())
            let events = model.snapshot?.recentEvents ?? model.latestEvents
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
