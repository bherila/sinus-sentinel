/// A fake `TrainingEngineProtocol`. `@unchecked Sendable` because the real
/// protocol has to cross into `finishTake`'s `Task.detached` (see
/// `TrainingEngineProtocol`'s doc comment); these tests never exercise that
/// path concurrently, so plain mutable state is fine.
private final class MockTrainingEngine: TrainingEngineProtocol, @unchecked Sendable {
    var trainingSnapshot = makeEmptyTrainingSnapshot()
    var bulkResult: Result<AppleBulkFeedbackResult, Error> = .success(
        AppleBulkFeedbackResult(groupsChanged: 1, classifierChanged: true, syncRequired: true)
    )

    func training() throws -> TrainingSnapshot { trainingSnapshot }
    func beginTeachTake() throws {}
    func cancelTeachTake() throws {}
    func enrollTake(eventType: AppleEventType, samples: [Float]) throws -> TeachResult {
        TeachResult(eventType: eventType, examples: 1, similarity: 0.9, separation: 0.5, peakDbfs: nil, good: true)
    }
    func deleteTake(id: Int64) throws -> UInt32 { 0 }
    func deleteClassTraining(eventType: AppleEventType) throws -> UInt32 { 0 }
    func deleteLearnedSuppressions() throws -> UInt32 { 0 }
    func deleteAllTraining() throws -> UInt32 { 0 }
    func removeTrainingGroup(groupId: String) throws -> AppleBulkFeedbackResult { try bulkResult.get() }
    func removeAllFeedbackTraining() throws -> AppleBulkFeedbackResult { try bulkResult.get() }
}

/// `AudioMonitoringService` only needs *some* `AppleEngine` to store — it
/// never calls into it unless a take is actually recording, which these
/// tests never start. `AppleEngine(noHandle:)` is the fake constructor
/// UniFFI generates for exactly this: an object with no real Rust handle.
private func makeFakeAudio() -> AudioMonitoringService {
    AudioMonitoringService(engine: AppleEngine(noHandle: .init()))
}

private func makeGroup(id: String = "g1") -> TrainingGroup {
    TrainingGroup(
        groupId: id,
        provenance: .guidedTake,
        eventType: .cough,
        originalEventType: nil,
        createdAt: "now",
        peakDbfs: nil,
        modelVersion: nil,
        synced: false
    )
}

@MainActor
func testTrainingModel() {
    // A good take is a durable outbound change: it must ask for a sync.
    do {
        let model = TrainingModel()
        var changedCount = 0
        model.onTrainingChanged = { changedCount += 1 }

        let result = TeachResult(eventType: .cough, examples: 3, similarity: 0.92, separation: 0.4, peakDbfs: nil, good: true)
        model.handleSaved(result)

        expect(changedCount == 1, "handleSaved calls onTrainingChanged")
        expect(model.message?.contains("Saved Cough sample #3") == true, "handleSaved message names the class and take count")
    }

    // removeTrainingGroup only requests a sync when the result says one is required.
    do {
        let engine = MockTrainingEngine()
        let model = TrainingModel()
        model.attach(engine: engine, audio: makeFakeAudio(), modelReady: true)
        var changedCount = 0
        model.onTrainingChanged = { changedCount += 1 }

        engine.bulkResult = .success(AppleBulkFeedbackResult(groupsChanged: 1, classifierChanged: false, syncRequired: false))
        model.removeTrainingGroup(makeGroup())
        expect(changedCount == 0, "removeTrainingGroup(syncRequired: false) does not request a sync")

        engine.bulkResult = .success(AppleBulkFeedbackResult(groupsChanged: 1, classifierChanged: false, syncRequired: true))
        model.removeTrainingGroup(makeGroup())
        expect(changedCount == 1, "removeTrainingGroup(syncRequired: true) requests a sync")
    }

    // removeAllFeedbackTraining's message pluralizes on the changed count.
    do {
        let engine = MockTrainingEngine()
        let model = TrainingModel()
        model.attach(engine: engine, audio: makeFakeAudio(), modelReady: true)

        engine.bulkResult = .success(AppleBulkFeedbackResult(groupsChanged: 0, classifierChanged: false, syncRequired: false))
        model.removeAllFeedbackTraining()
        expect(model.message == "There was no feedback-derived training to remove.", "removeAllFeedbackTraining(0)")

        engine.bulkResult = .success(AppleBulkFeedbackResult(groupsChanged: 1, classifierChanged: true, syncRequired: true))
        model.removeAllFeedbackTraining()
        expect(model.message == "Removed feedback-derived training from 1 event.", "removeAllFeedbackTraining(1) is singular")

        engine.bulkResult = .success(AppleBulkFeedbackResult(groupsChanged: 3, classifierChanged: true, syncRequired: true))
        model.removeAllFeedbackTraining()
        expect(model.message == "Removed feedback-derived training from 3 events.", "removeAllFeedbackTraining(3) is plural")
    }
}
