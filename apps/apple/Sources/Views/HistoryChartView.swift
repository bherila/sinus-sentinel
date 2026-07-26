import Charts
import SwiftUI

struct HistoryChartView: View {
    let snapshot: HistorySnapshot?

    private var points: [HistoryPoint] {
        snapshot?.days.flatMap { day in
            day.counts.map { count in
                HistoryPoint(
                    date: day.dateIso8601,
                    eventType: count.eventType,
                    count: count.count
                )
            }
        } ?? []
    }

    private var hasEvents: Bool {
        points.contains { $0.count > 0 }
    }

    var body: some View {
        Group {
            if !hasEvents {
                ContentUnavailableView(
                    "No events this week",
                    systemImage: "chart.bar",
                    description: Text("Detected events will appear here.")
                )
            } else {
                Chart(points) { point in
                    BarMark(
                        x: .value("Day", point.date),
                        y: .value("Events", point.count),
                        stacking: .standard
                    )
                    .foregroundStyle(by: .value("Type", point.eventType.displayName))
                    .accessibilityLabel("\(point.eventType.displayName), \(point.date)")
                    .accessibilityValue("\(point.count) events")
                }
                // Domain is the full taxonomy, in fixed order — not whatever
                // subset of classes happened to occur this week. `foregroundStyle(by:)`
                // otherwise assigns colors from the data present, so a week
                // without sneezes would shift every other class's color.
                .chartForegroundStyleScale(
                    domain: AppleEventType.allCases.map(\.displayName),
                    range: AppleEventType.allCases.map(\.color)
                )
                .chartLegend(position: .bottom, alignment: .leading)
            }
        }
        .frame(minHeight: 220)
    }
}

private struct HistoryPoint: Identifiable {
    let date: String
    let eventType: AppleEventType
    let count: UInt64

    var id: String { "\(date)-\(eventType.displayName)" }
}

extension AppleEventType {
    var displayName: String {
        switch self {
        case .cough: "Cough"
        case .throatClearing: "Throat clearing"
        case .sniffle: "Sniffle"
        case .sneeze: "Sneeze"
        case .noseBlow: "Nose blow"
        case .hawk: "Hawk"
        case .snortSuck: "Snort / suck"
        }
    }

    /// Bound to the class, not derived from the data — see the fixed
    /// `chartForegroundStyleScale` domain above.
    var color: Color {
        switch self {
        case .cough: .blue
        case .throatClearing: .green
        case .sniffle: .pink
        case .sneeze: .orange
        case .noseBlow: .teal
        case .hawk: .red
        case .snortSuck: .purple
        }
    }
}
