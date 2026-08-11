/// A fake `HistoryEngineProtocol` that records which method fired and
/// returns a canned result/error, so `testHistoryModel` can drive
/// `HistoryModel` without a real `AppleEngine`.
private final class MockHistoryEngine: HistoryEngineProtocol {
    var feedbackResult: Result<AppleFeedbackResult, Error> = .success(makeFeedbackResult())
    var historySnapshot = makeEmptyHistorySnapshot()
    var historyCallCount = 0
    var calledMethods: [String] = []

    func history(days: UInt32, nowEpochMs: Int64, timezoneOffsetMinutes: Int32) throws -> HistorySnapshot {
        historyCallCount += 1
        return historySnapshot
    }

    func reportFalsePositive(eventUuid: String) throws -> AppleFeedbackResult {
        calledMethods.append("reportFalsePositive")
        return try feedbackResult.get()
    }

    func confirmEvent(eventUuid: String) throws -> AppleFeedbackResult {
        calledMethods.append("confirmEvent")
        return try feedbackResult.get()
    }

    func recharacterize(eventUuid: String, corrected: AppleEventType) throws -> AppleFeedbackResult {
        calledMethods.append("recharacterize")
        return try feedbackResult.get()
    }

    func clearFlag(eventUuid: String) throws -> AppleFeedbackResult {
        calledMethods.append("clearFlag")
        return try feedbackResult.get()
    }
}

@MainActor
func testHistoryModel() {
    // refresh() runs only when the result says the event changed.
    do {
        let engine = MockHistoryEngine()
        let model = HistoryModel()
        model.attach(engine: engine)

        engine.feedbackResult = .success(makeFeedbackResult(eventChanged: false))
        model.reportFalsePositive(makeEvent())
        expect(engine.historyCallCount == 0, "eventChanged=false does not refresh")

        engine.feedbackResult = .success(makeFeedbackResult(eventChanged: true))
        model.reportFalsePositive(makeEvent())
        expect(engine.historyCallCount == 1, "eventChanged=true refreshes")
    }

    // onSyncRequired fires only when the result says a sync is required —
    // including the device-local confirm case (embedding scored locally, no
    // change the PHR needs to hear about).
    do {
        let engine = MockHistoryEngine()
        let model = HistoryModel()
        model.attach(engine: engine)
        var syncCount = 0
        model.onSyncRequired = { syncCount += 1 }

        engine.feedbackResult = .success(makeFeedbackResult(syncRequired: false))
        model.confirm(makeEvent())
        expect(syncCount == 0, "confirm with syncRequired=false does not request a sync")

        engine.feedbackResult = .success(makeFeedbackResult(syncRequired: true))
        model.confirm(makeEvent())
        expect(syncCount == 1, "confirm with syncRequired=true requests a sync")
    }

    // Each action produces the right message family.
    do {
        let engine = MockHistoryEngine()
        let model = HistoryModel()
        model.attach(engine: engine)

        engine.feedbackResult = .success(makeFeedbackResult(effect: .applied))
        model.reportFalsePositive(makeEvent(eventType: .cough))
        expect(
            model.message == "Reported the Cough: it no longer counts here or in the PHR, and detection was updated from this event.",
            "reportFalsePositive uses the report message family"
        )

        model.confirm(makeEvent(eventType: .sneeze))
        expect(
            model.message == "Confirmed the Sneeze and updated detection immediately.",
            "confirm uses the confirm message family"
        )

        model.recharacterize(makeEvent(eventType: .cough), to: .sneeze)
        expect(
            model.message == "Corrected to Sneeze and updated detection immediately.",
            "recharacterize uses the correct message family"
        )

        engine.feedbackResult = .success(makeFeedbackResult(eventChanged: true))
        model.clearFlag(makeEvent())
        expect(
            model.message == "Restored the original event and removed the training derived from this feedback.",
            "clearFlag uses the undo message family"
        )
    }

    // A stale uuid produces the distinct NotFound message and still
    // refreshes, so the row that no longer exists disappears from the list.
    do {
        let engine = MockHistoryEngine()
        let model = HistoryModel()
        model.attach(engine: engine)
        engine.feedbackResult = .failure(AppleEngineError.NotFound(uuid: "missing"))

        model.reportFalsePositive(makeEvent())
        expect(model.message == "That event is no longer on this device.", "NotFound produces the stale-row message")
        expect(engine.historyCallCount == 1, "NotFound still refreshes")
    }

    // Recharacterizing back to the class the detector originally recorded is
    // a client-side undo shortcut — it must route through clearFlag rather
    // than calling Rust's recharacterize, matching flag::recharacterize's own
    // rule, so the message comes out in the undo family.
    do {
        let engine = MockHistoryEngine()
        let model = HistoryModel()
        model.attach(engine: engine)
        engine.feedbackResult = .success(makeFeedbackResult(eventChanged: true))

        model.recharacterize(makeEvent(eventType: .cough), to: .cough)
        expect(engine.calledMethods == ["clearFlag"], "recharacterize to the original class routes through clearFlag")

        // On an already-corrected event (original cough, shown as sneeze),
        // picking cough is the undo; picking a third class is a correction.
        engine.calledMethods = []
        model.recharacterize(makeEvent(eventType: .sneeze, originalEventType: .cough), to: .cough)
        expect(engine.calledMethods == ["clearFlag"], "the original class stays the undo even after a correction")

        engine.calledMethods = []
        model.recharacterize(makeEvent(eventType: .sneeze, originalEventType: .cough), to: .hawk)
        expect(engine.calledMethods == ["recharacterize"], "a third class is a real correction")
    }
}
