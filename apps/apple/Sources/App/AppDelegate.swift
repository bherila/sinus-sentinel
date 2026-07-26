#if os(macOS)
import AppKit

/// `LSUIElement` puts the process in `.accessory` activation policy: a menu-bar
/// item, no Dock icon — and no application menu. `Settings…` and its ⌘, live in
/// the application menu, so under `.accessory` the shortcut has nowhere to attach
/// and silently does nothing. The fix is to make the activation policy track
/// whether any ordinary window is open, switching to `.regular` (which restores
/// the application menu) right before such a window is shown.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(windowWillClose(_:)),
            name: NSWindow.willCloseNotification,
            object: nil
        )

        // SwiftUI restores a `Window` scene's state across launches, but the
        // window does not exist yet while this callback runs — it is created on
        // a later turn of the run loop. Deferring the check lets it show up.
        DispatchQueue.main.async {
            if Self.hasOrdinaryWindow() {
                NSApp.setActivationPolicy(.regular)
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    /// Raises the app to `.regular` and activates it. Call this immediately
    /// before opening a window — never in response to mere activation, since
    /// toggling the policy while the app simply becomes active makes the
    /// menu-bar item flicker.
    static func activateForWindow() {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func windowWillClose(_ notification: Notification) {
        guard let closing = notification.object as? NSWindow else { return }
        if !Self.hasOrdinaryWindow(excluding: closing) {
            NSApp.setActivationPolicy(.accessory)
        }
    }

    /// "Ordinary" means `canBecomeMain && isVisible`. The `MenuBarExtra` status
    /// item is backed by a window that can never become main, so this predicate
    /// excludes it without needing to filter by title or class.
    private static func hasOrdinaryWindow(excluding excluded: NSWindow? = nil) -> Bool {
        NSApp.windows.contains { window in
            window !== excluded && window.canBecomeMain && window.isVisible
        }
    }
}
#endif
