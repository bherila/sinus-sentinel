import Foundation
#if os(macOS)
import CoreGraphics
#endif

/// Time since the last keyboard/mouse/trackpad input, mirroring
/// `apps/desktop/src/power.rs::user_idle` so quiet hours reads the same signal
/// on every shell.
enum UserIdleMonitor {
    /// `nil` means "no honest idle signal", not "the user just touched the
    /// machine" — a query failure must fall back to the literal quiet-hours
    /// window (SPEC §13 q4), never to "definitely present".
    static func idle() -> TimeInterval? {
        #if os(macOS)
        // `kCGAnyInputEventType` is a `#define` cast, not a `CGEventType` case,
        // so ClangImporter does not surface it as a symbol — reconstruct it the
        // same way the header defines it: all bits set.
        let anyInputEventType = CGEventType(rawValue: UInt32.max)!
        let seconds = CGEventSource.secondsSinceLastEventType(.hidSystemState, eventType: anyInputEventType)
        return (seconds.isFinite && seconds >= 0) ? seconds : nil
        #else
        // iOS: the device is in the user's pocket, so there is no absence to
        // detect. Deriving a proxy from app-active or screen-lock state would
        // silently redefine quiet hours as "whenever your screen is off" —
        // `nil` correctly falls back to the literal window instead.
        return nil
        #endif
    }
}
