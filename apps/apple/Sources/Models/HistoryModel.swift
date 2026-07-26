import Foundation
import Observation
import SinusAppleFFI

@MainActor
@Observable
final class HistoryModel {
    private(set) var snapshot: HistorySnapshot?

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

    private static var nowMilliseconds: Int64 {
        Int64(Date().timeIntervalSince1970 * 1_000)
    }
}
