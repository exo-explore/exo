import AppKit

/// Brings one of EXO's windows (Settings, the bug report) to the front as the
/// active window.
///
/// EXO lives in the menu bar, and on macOS 27 choosing an item in its menu
/// doesn't make it the active app: `NSApp.activate()` is only a request, which
/// the active app turns down, so the window opens without keyboard focus,
/// sometimes behind the active app. While such a window is open, EXO is a
/// regular app instead (Dock icon, ⌘-Tab), as it is for the welcome window,
/// and goes back to the menu bar when the last of them closes.
@MainActor
enum WindowActivation {
    /// Windows that keep EXO a regular app while they are open.
    private static var windows: [ObjectIdentifier: NSObjectProtocol] = [:]

    static func bringToFront(_ window: NSWindow) {
        keepRegular(while: window)
        if let frontmost = NSWorkspace.shared.frontmostApplication,
            frontmost != NSRunningApplication.current
        {
            NSRunningApplication.current.activate(from: frontmost, options: [])
        }
        NSApp.activate()
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
    }

    private static func keepRegular(while window: NSWindow) {
        if NSApp.activationPolicy() != .regular {
            NSApp.setActivationPolicy(.regular)
        }
        let id = ObjectIdentifier(window)
        guard windows[id] == nil else { return }
        windows[id] = NotificationCenter.default.addObserver(
            forName: NSWindow.willCloseNotification, object: window, queue: .main
        ) { _ in
            MainActor.assumeIsolated { windowClosed(id) }
        }
    }

    private static func windowClosed(_ id: ObjectIdentifier) {
        if let observer = windows.removeValue(forKey: id) {
            NotificationCenter.default.removeObserver(observer)
        }
        // Back to the menu bar only, unless another window (such as the
        // first-run welcome) is still open.
        let otherWindowOpen = NSApp.windows.contains {
            $0.isVisible && $0.styleMask.contains(.titled) && ObjectIdentifier($0) != id
        }
        if windows.isEmpty, !otherWindowOpen {
            NSApp.setActivationPolicy(.accessory)
        }
    }
}
