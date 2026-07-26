import SwiftUI
import SinusAppleFFI

/// Settings › Training — the Teach-mode pane. Mirrors the desktop tray's
/// Teach mode section (`app.rs:797-993`) field-for-field, including its exact
/// status and feedback strings, so a user running both shells sees the same
/// thing.
struct TrainingSettingsView: View {
    @Environment(EngineHost.self) private var host
    @State private var pendingDeletion: PendingDeletion?

    private enum PendingDeletion: Equatable {
        case one(id: Int64, eventType: AppleEventType)
        case classAll(AppleEventType)
        case negatives
        case all
    }

    private var training: TrainingModel { host.training }
    private var busy: Bool { training.phase == .armed || training.phase == .recording }
    private var micRunning: Bool { host.monitor.isCapturing }
    private var canRecord: Bool { !busy && training.modelReady && micRunning }

    var body: some View {
        let snapshot = training.snapshot

        Form {
            Section {
                Text("Teach your own sounds. Raw audio is discarded; only an embedding is saved — and synced to the PHR, if connected, so other machines inherit your training.")
                Text("Record one clear sound after the short get-ready countdown. Add 3–5 varied takes per class.")
            }
            .font(.footnote)
            .foregroundStyle(.secondary)

            ForEach(snapshot?.classes ?? [], id: \.eventType) { classTraining in
                Section {
                    HStack {
                        Button("Record take") {
                            training.recordTake(classTraining.eventType)
                        }
                        .disabled(!canRecord)

                        Button("Reset class") {
                            pendingDeletion = .classAll(classTraining.eventType)
                        }
                        .disabled(busy || classTraining.takes.isEmpty)
                    }

                    ForEach(Array(classTraining.takes.enumerated()), id: \.element.id) { index, take in
                        HStack {
                            Text("Take \(index + 1) • \(formattedTakeTime(take.createdAt))")
                            Spacer()
                            if let similarity = take.similarity, let separation = take.separation {
                                Text("repeat \(String(format: "%.2f", similarity)) • separation \(String(format: "%+.2f", separation))")
                                    .foregroundStyle(.secondary)
                            } else {
                                Text("baseline take")
                                    .foregroundStyle(.secondary)
                            }
                            Button("Remove") {
                                pendingDeletion = .one(id: take.id, eventType: classTraining.eventType)
                            }
                            .disabled(busy)
                        }
                        .font(.footnote)
                    }
                } header: {
                    Text("\(classTraining.eventType.displayName) — \(statusText(classTraining.status, count: classTraining.takes.count))")
                }
            }

            if let snapshot, snapshot.negativeCount > 0 {
                Section {
                    HStack {
                        Text(
                            "\(snapshot.negativeCount) reported false positive\(snapshot.negativeCount == 1 ? " is" : "s are") teaching the detector what to ignore."
                        )
                        Spacer()
                        Button("Forget reports") {
                            pendingDeletion = .negatives
                        }
                        .disabled(busy)
                    }
                }
            }

            Section {
                Button("Reset all training") {
                    pendingDeletion = .all
                }
                .disabled(busy || !hasAnyTraining(snapshot))

                if !training.modelReady {
                    Text("Teach mode needs the YAMNet model; restart after fixing a \u{2018}model missing\u{2019} status.")
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }
                if !micRunning {
                    Text("Start a monitoring session to record a take — Teach mode needs the microphone live to buffer one.")
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }
                if let message = training.message {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(feedbackColor)
                }
            }
        }
        .formStyle(.grouped)
        .onAppear {
            // Nothing but this pane changes training, so there is no timer —
            // just refresh whenever it becomes visible.
            training.refresh()
        }
        .confirmationDialog(
            pendingDeletion.map(confirmationTitle) ?? "",
            isPresented: Binding(
                get: { pendingDeletion != nil },
                set: { if !$0 { pendingDeletion = nil } }
            ),
            presenting: pendingDeletion
        ) { deletion in
            Button("Confirm removal", role: .destructive) {
                perform(deletion)
                pendingDeletion = nil
            }
            Button("Cancel", role: .cancel) {
                pendingDeletion = nil
            }
        } message: { _ in
            Text("Only local embeddings and their metadata will be removed; event history is unchanged.")
        }
    }

    private var feedbackColor: Color {
        switch training.phase {
        case .recording: .green
        case .failed: .red
        default: .secondary
        }
    }

    private func hasAnyTraining(_ snapshot: TrainingSnapshot?) -> Bool {
        guard let snapshot else { return false }
        return snapshot.classes.contains { !$0.takes.isEmpty } || snapshot.negativeCount > 0
    }

    private func statusText(_ status: TrainingStatus, count: Int) -> String {
        switch status {
        case .untrained: "not trained"
        case .inactive(let needed): "inactive • needs \(needed) more"
        case .ready: "ready • \(count) takes"
        case .active: "active • \(count) takes • add varied takes"
        }
    }

    private func confirmationTitle(_ deletion: PendingDeletion) -> String {
        switch deletion {
        case .one(_, let eventType):
            "Remove this \(eventType.displayName) take?"
        case .classAll(let eventType):
            "Reset every \(eventType.displayName) take?"
        case .negatives:
            "Forget every reported false positive? Previously suppressed sounds may be detected again."
        case .all:
            "Reset every saved Teach-mode take?"
        }
    }

    private func perform(_ deletion: PendingDeletion) {
        switch deletion {
        case .one(let id, _):
            training.deleteTake(id: id)
        case .classAll(let eventType):
            training.deleteClass(eventType)
        case .negatives:
            training.deleteLearnedSuppressions()
        case .all:
            training.deleteAllTraining()
        }
    }

    private func formattedTakeTime(_ createdAt: String) -> String {
        for formatter in Self.rfc3339Formatters {
            if let date = formatter.date(from: createdAt) {
                return Self.takeTimeFormatter.string(from: date)
            }
        }
        return "saved previously"
    }

    private static let rfc3339Formatters: [ISO8601DateFormatter] = [
        {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            return formatter
        }(),
        {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime]
            return formatter
        }(),
    ]

    private static let takeTimeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, h:mm a"
        return formatter
    }()
}
