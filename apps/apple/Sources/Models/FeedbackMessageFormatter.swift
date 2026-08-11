/// Which affordance produced an `AppleFeedbackResult` — used only to select
/// the right message family below. Mirrors the three branches
/// `apps/desktop/src/app.rs` already has (`report_false_positive`,
/// `recharacterize`, `clear_flag`) plus a confirm family the desktop tray
/// does not need yet, since it has no "already correct" affordance of its own.
enum FeedbackAction {
    case report, confirm, correct, undo
}

/// Turns one `AppleFeedbackResult`/`AppleBulkFeedbackResult` into the exact
/// copy the History and Training panes show. Pure — no engine, no SwiftUI —
/// so it can be shared by `HistoryModel`, `TrainingModel`, and their tests
/// instead of each writing its own `switch`. The report/correct/undo strings
/// must read identically to `apps/desktop/src/app.rs`'s
/// `report_false_positive`/`recharacterize`/`clear_flag` (~lines 261-379), so
/// a user running both shells on one machine sees the same wording.
enum FeedbackMessageFormatter {
    static func message(
        action: FeedbackAction,
        result: AppleFeedbackResult,
        originalClass: String,
        targetClass: String?
    ) -> String {
        switch action {
        case .report:
            return reportMessage(effect: result.effect, className: originalClass)
        case .confirm:
            return confirmMessage(effect: result.effect, progress: result.progress, className: originalClass)
        case .correct:
            // `targetClass` is always supplied by callers that pick `.correct`
            // (the correction target); `originalClass` is a defensive fallback
            // so this stays total rather than force-unwrapping.
            return correctMessage(effect: result.effect, progress: result.progress, now: targetClass ?? originalClass)
        case .undo:
            return result.eventChanged
                ? "Restored the original event and removed the training derived from this feedback."
                : "This event had no feedback to undo; detection was unchanged."
        }
    }

    private static func reportMessage(effect: AppleTrainingEffect, className: String) -> String {
        switch effect {
        case .applied:
            return "Reported the \(className): it no longer counts here or in the PHR, and detection was updated from this event."
        case .unavailable:
            return "Reported the \(className): it no longer counts here or in the PHR. No local embedding was available, so detection was not retrained."
        case .unchanged:
            return "This \(className) was already reported; no duplicate training was added."
        case .removed:
            return "Reported the \(className) and removed its obsolete training."
        }
    }

    private static func confirmMessage(
        effect: AppleTrainingEffect,
        progress: AppleTrainingProgress?,
        className: String
    ) -> String {
        switch effect {
        case .applied:
            if let progress, progress.positiveCount < progress.activationThreshold {
                let needed = progress.activationThreshold - progress.positiveCount
                return "Confirmed the \(className) — learning from this event: \(progress.positiveCount) of \(progress.activationThreshold) examples; \(needed) more needed."
            }
            return "Confirmed the \(className) and updated detection immediately."
        case .unavailable:
            return "Confirmed the \(className). No local embedding was available, so detection was not retrained."
        case .unchanged:
            return "This \(className) was already confirmed; no duplicate training was added."
        case .removed:
            // Unreachable by construction — confirming an event never removes
            // training — but this formatter stays total rather than trapping
            // on a case Rust could add later.
            return "Confirmed the \(className)."
        }
    }

    private static func correctMessage(
        effect: AppleTrainingEffect,
        progress: AppleTrainingProgress?,
        now: String
    ) -> String {
        switch effect {
        case .applied:
            if let progress, progress.positiveCount < progress.activationThreshold {
                let needed = progress.activationThreshold - progress.positiveCount
                return "Corrected to \(now) and learned from this event — \(progress.positiveCount) of \(progress.activationThreshold) examples; \(needed) more needed."
            }
            return "Corrected to \(now) and updated detection immediately."
        case .unavailable:
            return "Corrected to \(now), but this event no longer has a local embedding, so detection was not retrained."
        case .unchanged:
            return "This event was already corrected to \(now); no duplicate training was added."
        case .removed:
            return "Corrected to \(now) and removed obsolete training."
        }
    }

    /// One Training group's removal — the per-row action in Settings › Training.
    static func groupRemovalMessage(result: AppleBulkFeedbackResult) -> String {
        result.groupsChanged > 0
            ? "Removed this training unit; detection was updated."
            : "That training was already gone."
    }

    /// The bulk "remove all feedback-derived training" action. Unlike a
    /// single group, the count is user-visible, so it needs its own noun.
    static func bulkRemovalMessage(result: AppleBulkFeedbackResult) -> String {
        guard result.groupsChanged > 0 else {
            return "There was no feedback-derived training to remove."
        }
        let noun = result.groupsChanged == 1 ? "event" : "events"
        return "Removed feedback-derived training from \(result.groupsChanged) \(noun)."
    }
}
