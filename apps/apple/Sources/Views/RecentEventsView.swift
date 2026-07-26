import SwiftUI

/// Mirrors `app.rs`'s "Recent events" list: detail per row plus the three
/// flag actions, and the "heard something" indicator while the gate is open.
struct RecentEventsView: View {
    @Environment(EngineHost.self) private var host

    private var events: [AppleEvent] {
        host.history.snapshot?.recentEvents ?? []
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent")
                .font(.title2.bold())

            if host.monitor.status?.gateOpen == true {
                Text("heard something at \(heardTimeText) — classifying…")
                    .foregroundStyle(.blue)
            }

            if events.isEmpty {
                Text("No recent events")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(events, id: \.uuid) { event in
                    RecentEventRow(event: event)
                }
            }

            if let message = host.history.message {
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var heardTimeText: String {
        let epochMs = host.monitor.status?.lastHeardEpochMs
        let date = epochMs.map { Date(timeIntervalSince1970: Double($0) / 1_000) } ?? Date()
        return Self.timeFormatter.string(from: date)
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()
}

private struct RecentEventRow: View {
    @Environment(EngineHost.self) private var host
    let event: AppleEvent

    /// Both a report and a correction are undoable — a correction is just as
    /// much a user judgement that can be wrong.
    private var isUndoable: Bool {
        event.falsePositive || event.correctedTo != nil
    }

    private var recharacterizeOptions: [AppleEventType] {
        AppleEventType.allCases.filter { $0 != event.eventType }
    }

    var body: some View {
        HStack(spacing: 8) {
            if isUndoable {
                Button {
                    host.history.clearFlag(event)
                } label: {
                    Image(systemName: "arrow.uturn.backward")
                }
                .help(
                    event.falsePositive
                        ? "Undo: count this event again, here and in the PHR"
                        : "Undo the correction, here and in the PHR"
                )
            }

            if !event.falsePositive {
                Button {
                    host.history.reportFalsePositive(event)
                } label: {
                    Image(systemName: "xmark")
                }
                .help(
                    "Report false positive: stops this counting (here and in the PHR) and teaches the detector not to label that sound this way"
                )
            }

            Text("\(primaryText)\(Text(detailText).foregroundStyle(.secondary))")
                .strikethrough(event.falsePositive)
                .opacity(event.falsePositive ? 0.6 : 1.0)

            Spacer()

            if !event.falsePositive {
                Menu {
                    // The prompt is a section header rather than the button's
                    // label: it belongs to the list of classes, and repeating it
                    // on every row would crowd out the event it describes.
                    Section("Actually this was:") {
                        ForEach(recharacterizeOptions, id: \.self) { type in
                            Button(type.displayName) {
                                host.history.recharacterize(event, to: type)
                            }
                        }
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .help("Recharacterize: record what this sound really was")
                .fixedSize()
            }
        }
        .padding(.vertical, 4)
    }

    private var primaryText: String {
        let time = Self.dateFormatter.string(
            from: Date(timeIntervalSince1970: Double(event.occurredAtEpochMs) / 1_000)
        )
        var text = "\(time)  \(event.eventType.displayName)"
        if event.correctedTo != nil {
            text += " (was \(event.originalEventType.displayName))"
        }
        return text
    }

    private var detailText: String {
        var text = "  conf \(String(format: "%.2f", event.confidence))  x\(event.burstCount)"
        if let peak = event.peakDbfs {
            text += "  \(String(format: "%.0f", peak)) dBFS"
        }
        return text
    }

    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "MM-dd HH:mm:ss"
        return formatter
    }()
}
