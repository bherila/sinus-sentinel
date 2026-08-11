/// Shared fixture builders for the model tests. Kept in one place so a test
/// only has to override the field it actually cares about.

func makeEvent(
    eventType: AppleEventType = .cough,
    originalEventType: AppleEventType = .cough
) -> AppleEvent {
    AppleEvent(
        uuid: "test-uuid",
        eventType: eventType,
        originalEventType: originalEventType,
        correctedTo: nil,
        occurredAtEpochMs: 0,
        timezoneOffsetMinutes: 0,
        durationMs: 0,
        confidence: 1,
        burstCount: 1,
        peakDbfs: nil,
        meanDbfs: nil,
        noiseFloorDbfs: nil,
        modelVersion: "test",
        falsePositive: false,
        synced: false,
        confirmed: false
    )
}

func makeFeedbackResult(
    event: AppleEvent = makeEvent(),
    eventChanged: Bool = true,
    classifierChanged: Bool = false,
    syncRequired: Bool = true,
    effect: AppleTrainingEffect = .applied,
    progress: AppleTrainingProgress? = nil
) -> AppleFeedbackResult {
    AppleFeedbackResult(
        event: event,
        eventChanged: eventChanged,
        classifierChanged: classifierChanged,
        syncRequired: syncRequired,
        effect: effect,
        progress: progress
    )
}

func makeEmptyHistorySnapshot() -> HistorySnapshot {
    HistorySnapshot(
        today: [],
        days: [],
        recentEvents: [],
        congestionScorePerMonitoredHour: 0,
        monitoredHours: 0
    )
}

func makeEmptyTrainingSnapshot() -> TrainingSnapshot {
    TrainingSnapshot(classes: [], negativeCount: 0, groups: [])
}
