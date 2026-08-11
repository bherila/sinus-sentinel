/// Every action x effect combination `FeedbackMessageFormatter` is specified
/// to produce, asserted against the exact strings — a typo here is a typo a
/// user reads, and the desktop shell must read the same (report/correct/undo
/// families mirror `apps/desktop/src/app.rs`).
func testFeedbackMessageFormatter() {
    // report
    expect(
        FeedbackMessageFormatter.message(action: .report, result: makeFeedbackResult(effect: .applied), originalClass: "cough", targetClass: nil)
            == "Reported the cough: it no longer counts here or in the PHR, and detection was updated from this event.",
        "report + applied"
    )
    expect(
        FeedbackMessageFormatter.message(action: .report, result: makeFeedbackResult(effect: .unavailable), originalClass: "cough", targetClass: nil)
            == "Reported the cough: it no longer counts here or in the PHR. No local embedding was available, so detection was not retrained.",
        "report + unavailable"
    )
    expect(
        FeedbackMessageFormatter.message(action: .report, result: makeFeedbackResult(effect: .unchanged), originalClass: "cough", targetClass: nil)
            == "This cough was already reported; no duplicate training was added.",
        "report + unchanged"
    )
    expect(
        FeedbackMessageFormatter.message(action: .report, result: makeFeedbackResult(effect: .removed), originalClass: "cough", targetClass: nil)
            == "Reported the cough and removed its obsolete training.",
        "report + removed"
    )

    // confirm
    let confirmProgress = AppleTrainingProgress(eventType: .cough, positiveCount: 1, activationThreshold: 3)
    expect(
        FeedbackMessageFormatter.message(action: .confirm, result: makeFeedbackResult(effect: .applied, progress: confirmProgress), originalClass: "cough", targetClass: nil)
            == "Confirmed the cough — learning from this event: 1 of 3 examples; 2 more needed.",
        "confirm + applied, below threshold"
    )
    expect(
        FeedbackMessageFormatter.message(action: .confirm, result: makeFeedbackResult(effect: .applied, progress: nil), originalClass: "cough", targetClass: nil)
            == "Confirmed the cough and updated detection immediately.",
        "confirm + applied, no progress"
    )
    let confirmAtThreshold = AppleTrainingProgress(eventType: .cough, positiveCount: 3, activationThreshold: 3)
    expect(
        FeedbackMessageFormatter.message(action: .confirm, result: makeFeedbackResult(effect: .applied, progress: confirmAtThreshold), originalClass: "cough", targetClass: nil)
            == "Confirmed the cough and updated detection immediately.",
        "confirm + applied, at threshold"
    )
    expect(
        FeedbackMessageFormatter.message(action: .confirm, result: makeFeedbackResult(effect: .unavailable), originalClass: "cough", targetClass: nil)
            == "Confirmed the cough. No local embedding was available, so detection was not retrained.",
        "confirm + unavailable"
    )
    expect(
        FeedbackMessageFormatter.message(action: .confirm, result: makeFeedbackResult(effect: .unchanged), originalClass: "cough", targetClass: nil)
            == "This cough was already confirmed; no duplicate training was added.",
        "confirm + unchanged"
    )
    expect(
        FeedbackMessageFormatter.message(action: .confirm, result: makeFeedbackResult(effect: .removed), originalClass: "cough", targetClass: nil)
            == "Confirmed the cough.",
        "confirm + removed (unreachable, but total)"
    )

    // correct
    let correctProgress = AppleTrainingProgress(eventType: .sneeze, positiveCount: 2, activationThreshold: 5)
    expect(
        FeedbackMessageFormatter.message(action: .correct, result: makeFeedbackResult(effect: .applied, progress: correctProgress), originalClass: "cough", targetClass: "sneeze")
            == "Corrected to sneeze and learned from this event — 2 of 5 examples; 3 more needed.",
        "correct + applied, below threshold"
    )
    expect(
        FeedbackMessageFormatter.message(action: .correct, result: makeFeedbackResult(effect: .applied, progress: nil), originalClass: "cough", targetClass: "sneeze")
            == "Corrected to sneeze and updated detection immediately.",
        "correct + applied, no progress"
    )
    expect(
        FeedbackMessageFormatter.message(action: .correct, result: makeFeedbackResult(effect: .unavailable), originalClass: "cough", targetClass: "sneeze")
            == "Corrected to sneeze, but this event no longer has a local embedding, so detection was not retrained.",
        "correct + unavailable"
    )
    expect(
        FeedbackMessageFormatter.message(action: .correct, result: makeFeedbackResult(effect: .unchanged), originalClass: "cough", targetClass: "sneeze")
            == "This event was already corrected to sneeze; no duplicate training was added.",
        "correct + unchanged"
    )
    expect(
        FeedbackMessageFormatter.message(action: .correct, result: makeFeedbackResult(effect: .removed), originalClass: "cough", targetClass: "sneeze")
            == "Corrected to sneeze and removed obsolete training.",
        "correct + removed"
    )

    // undo
    expect(
        FeedbackMessageFormatter.message(action: .undo, result: makeFeedbackResult(eventChanged: true), originalClass: "cough", targetClass: nil)
            == "Restored the original event and removed the training derived from this feedback.",
        "undo, eventChanged"
    )
    expect(
        FeedbackMessageFormatter.message(action: .undo, result: makeFeedbackResult(eventChanged: false), originalClass: "cough", targetClass: nil)
            == "This event had no feedback to undo; detection was unchanged.",
        "undo, !eventChanged"
    )

    // grouped removal
    expect(
        FeedbackMessageFormatter.groupRemovalMessage(result: AppleBulkFeedbackResult(groupsChanged: 1, classifierChanged: true, syncRequired: true))
            == "Removed this training unit; detection was updated.",
        "groupRemovalMessage, changed"
    )
    expect(
        FeedbackMessageFormatter.groupRemovalMessage(result: AppleBulkFeedbackResult(groupsChanged: 0, classifierChanged: false, syncRequired: false))
            == "That training was already gone.",
        "groupRemovalMessage, unchanged"
    )

    // bulk removal, singular/plural
    expect(
        FeedbackMessageFormatter.bulkRemovalMessage(result: AppleBulkFeedbackResult(groupsChanged: 0, classifierChanged: false, syncRequired: false))
            == "There was no feedback-derived training to remove.",
        "bulkRemovalMessage, none"
    )
    expect(
        FeedbackMessageFormatter.bulkRemovalMessage(result: AppleBulkFeedbackResult(groupsChanged: 1, classifierChanged: true, syncRequired: true))
            == "Removed feedback-derived training from 1 event.",
        "bulkRemovalMessage, singular"
    )
    expect(
        FeedbackMessageFormatter.bulkRemovalMessage(result: AppleBulkFeedbackResult(groupsChanged: 3, classifierChanged: true, syncRequired: true))
            == "Removed feedback-derived training from 3 events.",
        "bulkRemovalMessage, plural"
    )
}
