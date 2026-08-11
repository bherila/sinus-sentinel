import SwiftUI

/// Mirrors `app.rs`'s "Recent events" list: detail per row plus the flag
/// actions, and the "heard something" indicator while the gate is open.
struct RecentEventsView: View {
    @Environment(EngineHost.self) private var host

    private func events(at date: Date) -> [AppleEvent] {
        let cutoff = Int64(date.addingTimeInterval(-5 * 60).timeIntervalSince1970 * 1_000)
        return (host.history.snapshot?.recentEvents ?? []).filter {
            $0.occurredAtEpochMs >= cutoff
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent")
                .font(.title2.bold())

            if host.monitor.status?.gateOpen == true {
                Text("heard something at \(heardTimeText) — classifying…")
                    .foregroundStyle(.blue)
            }

            TimelineView(.periodic(from: .now, by: 1)) { context in
                let currentEvents = events(at: context.date)
                if currentEvents.isEmpty {
                    Text("No events in the last 5 minutes")
                        .foregroundStyle(.secondary)
                } else {
                    #if os(iOS)
                    // Swipe actions only exist on `List` rows; macOS keeps the
                    // ScrollView/LazyVStack below with its labeled buttons.
                    List(currentEvents, id: \.uuid) { event in
                        RecentEventRow(event: event)
                    }
                    .listStyle(.plain)
                    .frame(height: min(CGFloat(currentEvents.count) * 48, 260))
                    #else
                    ScrollView {
                        LazyVStack(spacing: 0) {
                            ForEach(currentEvents, id: \.uuid) { event in
                                RecentEventRow(event: event)
                            }
                        }
                    }
                    .frame(height: min(CGFloat(currentEvents.count) * 48, 260))
                    #endif
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

    /// A report, a correction, and a confirmation are all undoable — each is
    /// a user judgement recorded with one tap, and any of them can be a
    /// mistake worth taking back.
    private var isUndoable: Bool {
        event.falsePositive || event.correctedTo != nil || event.confirmed
    }

    /// Confirming endorses the event's current effective label, so it only
    /// makes sense while nothing else has already spoken for this event —
    /// the feedback states are mutually exclusive.
    private var canConfirm: Bool {
        !event.confirmed && !event.falsePositive && event.correctedTo == nil
    }

    /// Filters out the current effective type only. The original type stays
    /// selectable — picking it is the undo path, routed through `clearFlag`
    /// by `HistoryModel.recharacterize`, and that is intended.
    private var recharacterizeOptions: [AppleEventType] {
        AppleEventType.allCases.filter { $0 != event.eventType }
    }

    private var undoHelpText: String {
        if event.falsePositive {
            "Undo: count this event again, here and in the PHR"
        } else if event.correctedTo != nil {
            "Undo the correction, here and in the PHR"
        } else {
            "Undo the confirmation, here and in the PHR"
        }
    }

    var body: some View {
        HStack(spacing: 8) {
            #if os(macOS)
            // macOS has room for discoverable, labeled affordances instead of
            // bare icons — iOS gets the same actions as swipe actions below.
            if canConfirm {
                Button {
                    host.history.confirm(event)
                } label: {
                    Label("Confirm", systemImage: "checkmark.seal")
                }
                .help("Confirm: endorse this event's current label, here and in the PHR")
            }

            if !event.falsePositive {
                Button(role: .destructive) {
                    host.history.reportFalsePositive(event)
                } label: {
                    Label("Not an event", systemImage: "xmark")
                }
                .help(
                    "Report false positive: stops this counting (here and in the PHR) and teaches the detector not to label that sound this way"
                )
            }

            if isUndoable {
                Button {
                    host.history.clearFlag(event)
                } label: {
                    Label("Undo", systemImage: "arrow.uturn.backward")
                }
                .help(undoHelpText)
            }
            #endif

            Text("\(primaryText)\(Text(detailText).foregroundStyle(.secondary))")
                .strikethrough(event.falsePositive)
                .opacity(event.falsePositive ? 0.6 : 1.0)

            Spacer()

            if event.confirmed {
                Image(systemName: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                    .frame(width: 18)
                    .help("Confirmed: you endorsed this event's current label")
                    .accessibilityLabel("Confirmed")
            }

            Image(systemName: event.synced ? "cloud.fill" : "arrow.triangle.2.circlepath")
                .foregroundStyle(event.synced ? .green : .secondary)
                .frame(width: 18)
                .help(
                    event.synced
                        ? "Uploaded to the PHR"
                        : "Waiting to upload the event or its latest change"
                )
                .accessibilityLabel(event.synced ? "Uploaded" : "Upload pending")

            #if os(macOS)
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
                    Label("Correct", systemImage: "ellipsis.circle")
                }
                .help("Recharacterize: record what this sound really was")
                .fixedSize()
            }
            #endif
        }
        .padding(.vertical, 4)
        #if os(iOS)
        .swipeActions(edge: .leading, allowsFullSwipe: true) {
            if canConfirm {
                Button {
                    host.history.confirm(event)
                } label: {
                    Label("Confirm", systemImage: "checkmark.seal")
                }
                .tint(.green)
            }
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            if !event.falsePositive {
                Button(role: .destructive) {
                    host.history.reportFalsePositive(event)
                } label: {
                    Label("Not an event", systemImage: "xmark")
                }
            }
            if isUndoable {
                Button {
                    host.history.clearFlag(event)
                } label: {
                    Label("Undo", systemImage: "arrow.uturn.backward")
                }
                .tint(.gray)
            }
        }
        .contextMenu {
            if !event.falsePositive {
                Section("Actually this was:") {
                    ForEach(recharacterizeOptions, id: \.self) { type in
                        Button(type.displayName) {
                            host.history.recharacterize(event, to: type)
                        }
                    }
                }
            }
        }
        #endif
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
