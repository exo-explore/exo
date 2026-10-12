import AppKit

extension NSAlert {
    /// Brings EXO to the front, then runs the alert modally. EXO is a menu bar
    /// app that is usually not the active app, so a plain `runModal()` can
    /// open the alert behind other apps' windows, where it goes unnoticed
    /// while it blocks EXO's menu.
    @MainActor
    @discardableResult
    func runModalInFront() -> NSApplication.ModalResponse {
        NSApp.activate(ignoringOtherApps: true)
        return runModal()
    }
}
