import Foundation
import Observation
import SinusAppleFFI

@MainActor
@Observable
final class HistoryModel {
    private(set) var snapshot: HistorySnapshot?
    /// Result of the last report/confirm/recharacterize/undo, for the History
    /// window to render under the list — mirrors `app.rs`'s `history_message`.
    private(set) var message: String?

    var onError: (String?) -> Void = { _ in }
    /// Raised only when a feedback operation actually requires a sync, so the
    /// host can request one without this model knowing `SyncModel` exists.
    /// Distinct from `eventChanged`/`classifierChanged`: a confirm with no
    /// embedding, for instance, changes nothing that needs to leave the
    /// device, so this must not fire for it.
    var onSyncRequired: () -> Void = {}

    private var engine: HistoryEngineProtocol?

    func attach(engine: HistoryEngineProtocol) {
        self.engine = engine
    }

    func refresh() {
        guard let engine else { return }
        do {
            snapshot = try engine.history(
                days: 7,
                nowEpochMs: Self.nowMilliseconds,
                timezoneOffsetMinutes: Int32(TimeZone.current.secondsFromGMT() / 60)
            )
        } catch {
            onError(error.localizedDescription)
        }
    }

    /// Report a misdetection. Wording mirrors `app.rs::report_false_positive`.
    func reportFalsePositive(_ event: AppleEvent) {
        guard let engine else { return }
        do {
            let result = try engine.reportFalsePositive(eventUuid: event.uuid)
            apply(result, action: .report, originalClass: event.eventType.displayName, targetClass: nil)
        } catch AppleEngineError.NotFound(_) {
            reportStale()
        } catch {
            message = "Could not flag the event: \(error.localizedDescription)"
        }
    }

    /// Confirm that the detector's current effective label was correct.
    func confirm(_ event: AppleEvent) {
        guard let engine else { return }
        do {
            let result = try engine.confirmEvent(eventUuid: event.uuid)
            apply(result, action: .confirm, originalClass: event.eventType.displayName, targetClass: nil)
        } catch AppleEngineError.NotFound(_) {
            reportStale()
        } catch {
            message = "Could not confirm the event: \(error.localizedDescription)"
        }
    }

    /// Record what a misdetected sound actually was. Choosing the class the
    /// detector originally recorded is an undo, not a correction — the same
    /// rule `sinus_app::flag::recharacterize` applies — and routing it to
    /// `clearFlag` here, as `app.rs:315-320` does, keeps the undo message
    /// written once instead of a "Corrected to…" that actually restored.
    func recharacterize(_ event: AppleEvent, to corrected: AppleEventType) {
        if corrected == event.originalEventType {
            clearFlag(event)
            return
        }
        guard let engine else { return }
        do {
            let result = try engine.recharacterize(eventUuid: event.uuid, corrected: corrected)
            apply(result, action: .correct, originalClass: event.eventType.displayName, targetClass: corrected.displayName)
        } catch AppleEngineError.NotFound(_) {
            reportStale()
        } catch {
            message = "Could not update the event: \(error.localizedDescription)"
        }
    }

    /// Undo a false-positive report or a correction. See `reportFalsePositive`
    /// for why the rules live on the Rust side.
    func clearFlag(_ event: AppleEvent) {
        guard let engine else { return }
        do {
            let result = try engine.clearFlag(eventUuid: event.uuid)
            apply(result, action: .undo, originalClass: event.eventType.displayName, targetClass: nil)
        } catch AppleEngineError.NotFound(_) {
            reportStale()
        } catch {
            message = "Could not restore the event: \(error.localizedDescription)"
        }
    }

    /// Common tail of every feedback action: format the message, refresh the
    /// list only when the row actually changed, and ask for a sync only when
    /// one is actually required. `classifierChanged` needs nothing here —
    /// matcher reload is Rust-internal.
    private func apply(_ result: AppleFeedbackResult, action: FeedbackAction, originalClass: String, targetClass: String?) {
        message = FeedbackMessageFormatter.message(
            action: action,
            result: result,
            originalClass: originalClass,
            targetClass: targetClass
        )
        if result.eventChanged {
            refresh()
        }
        if result.syncRequired {
            onSyncRequired()
        }
    }

    /// The uuid a feedback call was given no longer resolves to a row — a
    /// stale list, or a row the PHR sync removed between render and tap.
    /// Refresh anyway so that stale row disappears instead of sitting there
    /// looking actionable.
    private func reportStale() {
        message = "That event is no longer on this device."
        refresh()
    }

    private static var nowMilliseconds: Int64 {
        Int64(Date().timeIntervalSince1970 * 1_000)
    }
}
