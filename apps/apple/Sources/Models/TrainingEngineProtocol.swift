/// The slice of `AppleEngineProtocol` that `TrainingModel` calls. See
/// `HistoryEngineProtocol` for why this is narrower than the generated
/// protocol rather than reusing it directly.
///
/// `Sendable`, matching `AppleEngineProtocol`: `finishTake` hands the engine
/// into a `Task.detached` to keep a Core ML inference off the main actor, so
/// whatever satisfies this protocol — real or fake — has to cross that
/// boundary safely.
protocol TrainingEngineProtocol: AnyObject, Sendable {
    func training() throws -> TrainingSnapshot
    func beginTeachTake() throws
    func cancelTeachTake() throws
    func enrollTake(eventType: AppleEventType, samples: [Float]) throws -> TeachResult
    func deleteTake(id: Int64) throws -> UInt32
    func deleteClassTraining(eventType: AppleEventType) throws -> UInt32
    func deleteLearnedSuppressions() throws -> UInt32
    func deleteAllTraining() throws -> UInt32
    func removeTrainingGroup(groupId: String) throws -> AppleBulkFeedbackResult
    func removeAllFeedbackTraining() throws -> AppleBulkFeedbackResult
}

extension AppleEngine: TrainingEngineProtocol {}
