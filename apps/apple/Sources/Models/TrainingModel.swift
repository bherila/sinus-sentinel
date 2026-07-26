import Foundation
import Observation
import SinusAppleFFI

/// Mirrors `TeachState` in `apps/desktop/src/shared.rs`. `.idle` is "nothing
/// going on"; the countdown is its own phase, distinct from `.recording`,
/// because the pane has to say "get ready" before it can say "make the sound".
enum TeachPhase {
    case idle, armed, recording, saved, failed
}

/// Settings › Training. Mirrors the desktop tray's Teach mode section
/// (`app.rs:797-993`), including its exact status and feedback strings, so a
/// user running both shells sees the same thing.
@MainActor
@Observable
final class TrainingModel {
    private(set) var snapshot: TrainingSnapshot?
    private(set) var phase: TeachPhase = .idle
    private(set) var recordingClass: AppleEventType?
    private(set) var message: String?
    /// Whether a compiled Core ML model was found — see `EngineHost.modelRunner()`.
    /// Exposed (not just used internally) so the pane can show the same "model
    /// missing" warning the desktop renders.
    private(set) var modelReady = false

    var onError: (String?) -> Void = { _ in }
    /// Raised after training changes, so the host can request a sync. The tray
    /// app does the same after a deletion, and for a reason worth keeping: a
    /// removal that never reaches the PHR gets re-taught to this device on the
    /// next pull.
    var onTrainingChanged: () -> Void = {}

    private var engine: AppleEngine?
    private var audio: AudioMonitoringService?
    private var recorder: TakeRecorder?

    func attach(engine: AppleEngine, audio: AudioMonitoringService, modelReady: Bool) {
        self.engine = engine
        self.audio = audio
        self.modelReady = modelReady
        refresh()
    }

    func refresh() {
        guard let engine else { return }
        do {
            snapshot = try engine.training()
        } catch {
            onError(error.localizedDescription)
        }
    }

    func recordTake(_ eventType: AppleEventType) {
        guard let engine, let audio else { return }
        guard recorder == nil else {
            message = "A take is already recording."
            return
        }
        guard audio.isRunning else {
            message = "Start a monitoring session before recording a take — Teach mode needs the microphone live to buffer one."
            return
        }
        guard modelReady else {
            message = "Teach mode needs the YAMNet model; restart after fixing a \u{2018}model missing\u{2019} status."
            return
        }

        do {
            try engine.beginTeachTake()
        } catch {
            onError(error.localizedDescription)
            return
        }

        phase = .armed
        recordingClass = eventType
        message = "Get ready — \(eventType.displayName) recording starts in about one second…"

        let recorder = TakeRecorder(
            onRecordingStarted: { [weak self] in
                Task { @MainActor in
                    self?.phase = .recording
                    self?.message = "Recording \(eventType.displayName) now — make the sound once."
                }
            },
            onComplete: { [weak self] samples in
                Task { @MainActor in
                    self?.finishTake(engine: engine, eventType: eventType, samples: samples)
                }
            }
        )
        self.recorder = recorder
        // Installed only after `beginTeachTake` succeeds, so persistence is
        // already suppressed before the first sample can reach the recorder.
        audio.onSamples = { [weak recorder] samples in
            recorder?.push(samples)
        }
    }

    /// Abandon a take that will never complete — the microphone going away
    /// mid-take, or this model being torn down. Without this, a stuck take
    /// would leave event persistence suppressed forever.
    func cancelTake() {
        guard recorder != nil else { return }
        audio?.onSamples = nil
        recorder = nil
        try? engine?.cancelTeachTake()
        phase = .idle
        recordingClass = nil
        message = nil
    }

    func deleteTake(id: Int64) {
        guard let engine else { return }
        do {
            let deleted = try engine.deleteTake(id: id)
            reportDeletion(deleted, noun: "saved take")
        } catch {
            message = "Could not update training: \(error.localizedDescription)"
        }
    }

    func deleteClass(_ eventType: AppleEventType) {
        guard let engine else { return }
        do {
            let deleted = try engine.deleteClassTraining(eventType: eventType)
            reportDeletion(deleted, noun: "saved take")
        } catch {
            message = "Could not update training: \(error.localizedDescription)"
        }
    }

    func deleteLearnedSuppressions() {
        guard let engine else { return }
        do {
            let deleted = try engine.deleteLearnedSuppressions()
            reportDeletion(deleted, noun: "false-positive report")
        } catch {
            message = "Could not update training: \(error.localizedDescription)"
        }
    }

    func deleteAllTraining() {
        guard let engine else { return }
        do {
            let deleted = try engine.deleteAllTraining()
            reportDeletion(deleted, noun: "saved take")
        } catch {
            message = "Could not update training: \(error.localizedDescription)"
        }
    }

    private func reportDeletion(_ deleted: UInt32, noun: String) {
        phase = .idle
        recordingClass = nil
        message = "Removed \(deleted) \(noun)\(deleted == 1 ? "" : "s"). Detection updated immediately."
        refresh()
        onTrainingChanged()
    }

    private func finishTake(engine: AppleEngine, eventType: AppleEventType, samples: [Float]) {
        audio?.onSamples = nil
        recorder = nil

        // `enrollTake` holds the writer mutex across a Core ML inference; running
        // it on the main actor would freeze the UI for the length of that
        // inference. `engine` is `Sendable`, so it can cross straight into the
        // detached task.
        Task.detached { [weak self] in
            do {
                let result = try engine.enrollTake(eventType: eventType, samples: samples)
                await self?.handleSaved(result)
            } catch {
                await self?.handleFailed(eventType: eventType, error: error)
            }
        }
    }

    private func handleSaved(_ result: TeachResult) {
        phase = .saved
        recordingClass = result.eventType
        let className = result.eventType.displayName
        if result.similarity < 0 {
            message = "Saved the first \(className) sample. Add at least two more for validation."
        } else {
            let verdict = result.good ? "good" : "keep adding varied samples"
            message = "Saved \(className) sample #\(result.examples) — repeat similarity \(String(format: "%.2f", result.similarity)), class separation \(String(format: "%+.2f", result.separation)): \(verdict)."
        }
        refresh()
    }

    private func handleFailed(eventType: AppleEventType, error: Error) {
        phase = .failed
        recordingClass = eventType
        // The tray app stops at the advice. Keep the detail too: a buffer of the
        // wrong length is rejected by `enroll_take` with a message naming both
        // counts, and that is a Swift bug to fix rather than a noisy room.
        message = "Could not save the \(eventType.displayName) sample; try again in a quieter moment. (\(error.localizedDescription))"
        refresh()
    }
}

/// Buffers one Teach-mode take off the main actor: the audio processing queue
/// calls `push` synchronously per callback, and accumulating straight into
/// `@Observable` state from that queue is not safe — so the buffer lives in
/// its own locked, `Sendable` type instead.
private final class TakeRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var countdownRemaining: Int
    private let takeSampleCount: Int
    private var samples: [Float] = []
    private var finished = false

    private let onRecordingStarted: @Sendable () -> Void
    private let onComplete: @Sendable ([Float]) -> Void

    init(
        onRecordingStarted: @escaping @Sendable () -> Void,
        onComplete: @escaping @Sendable ([Float]) -> Void
    ) {
        countdownRemaining = Int(teachCountdownSamples())
        takeSampleCount = Int(teachTakeSamples())
        samples.reserveCapacity(takeSampleCount)
        self.onRecordingStarted = onRecordingStarted
        self.onComplete = onComplete
    }

    /// Called on the audio processing queue with each converted 16 kHz buffer.
    func push(_ input: [Float]) {
        var justStartedRecording = false
        var completedSamples: [Float]?

        lock.lock()
        if !finished {
            var remainder = input[...]
            if countdownRemaining > 0 {
                let discard = min(countdownRemaining, remainder.count)
                countdownRemaining -= discard
                remainder = remainder.dropFirst(discard)
                justStartedRecording = countdownRemaining == 0
            }
            if countdownRemaining == 0, !remainder.isEmpty {
                let room = takeSampleCount - samples.count
                let take = min(room, remainder.count)
                if take > 0 {
                    samples.append(contentsOf: remainder.prefix(take))
                }
                if samples.count >= takeSampleCount {
                    finished = true
                    completedSamples = samples
                }
            }
        }
        lock.unlock()

        if justStartedRecording {
            onRecordingStarted()
        }
        if let completedSamples {
            onComplete(completedSamples)
        }
    }
}
