import Foundation
import Observation
import SinusAppleFFI

@MainActor
@Observable
final class HistoryModel {
    private(set) var snapshot: HistorySnapshot?
    /// Result of the last report/recharacterize/undo, for the History window to
    /// render under the list — mirrors `app.rs`'s `history_message`.
    private(set) var message: String?

    var onError: (String?) -> Void = { _ in }

    private var engine: AppleEngine?

    func attach(engine: AppleEngine) {
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
            refresh()
            let className = result.event.originalEventType.displayName
            message = result.trained
                ? "Reported the \(className): it no longer counts here or in the PHR, and the detector will stop labelling that sound \(className)."
                : "Reported the \(className): it no longer counts here or in the PHR. No embedding was stored for it, so the detector was not adjusted."
        } catch {
            message = "Could not flag the event: \(error.localizedDescription)"
        }
    }

    /// Record what a misdetected sound actually was. Choosing the classifier's
    /// original answer is an undo, not a correction — routed to `clearFlag`
    /// exactly as `app.rs:290-293` does, so the undo message is written once.
    func recharacterize(_ event: AppleEvent, to corrected: AppleEventType) {
        if corrected == event.originalEventType {
            clearFlag(event)
            return
        }
        guard let engine else { return }
        do {
            _ = try engine.recharacterize(eventUuid: event.uuid, corrected: corrected)
            refresh()
            let was = event.originalEventType.displayName
            let now = corrected.displayName
            message = "Recorded as \(now) instead of \(was) — it now counts as \(now) here and in the PHR, and the detector will stop calling that sound \(was). Teach \(now) a few more times for it to be recognised on its own."
        } catch {
            message = "Could not update the event: \(error.localizedDescription)"
        }
    }

    /// Undo a false-positive report or a correction. See `reportFalsePositive`
    /// for why the rules live on the Rust side.
    func clearFlag(_ event: AppleEvent) {
        guard let engine else { return }
        do {
            _ = try engine.clearFlag(eventUuid: event.uuid)
            refresh()
            message = "Restored the event here and in the PHR. Any training it produced is kept — use Settings › Teach mode to remove that too."
        } catch {
            message = "Could not restore the event: \(error.localizedDescription)"
        }
    }

    private static var nowMilliseconds: Int64 {
        Int64(Date().timeIntervalSince1970 * 1_000)
    }
}
