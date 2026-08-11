/// The slice of `AppleEngineProtocol` that `HistoryModel` calls. Narrower
/// than the generated protocol on purpose: a test double only has to
/// implement five methods instead of the whole engine surface, and
/// `HistoryModel` cannot accidentally reach for something it should not.
protocol HistoryEngineProtocol: AnyObject {
    func history(days: UInt32, nowEpochMs: Int64, timezoneOffsetMinutes: Int32) throws -> HistorySnapshot
    func reportFalsePositive(eventUuid: String) throws -> AppleFeedbackResult
    func confirmEvent(eventUuid: String) throws -> AppleFeedbackResult
    func recharacterize(eventUuid: String, corrected: AppleEventType) throws -> AppleFeedbackResult
    func clearFlag(eventUuid: String) throws -> AppleFeedbackResult
}

extension AppleEngine: HistoryEngineProtocol {}
