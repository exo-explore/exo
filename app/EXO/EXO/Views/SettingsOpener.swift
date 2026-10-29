import AppKit

/// Opens EXO's Settings window, from the menu (Settings… ⌘,) or when EXO.app
/// is opened again while it runs. One seam, so moving Settings to SwiftUI's
/// `Settings` scene only changes `openSettings`.
@MainActor
enum SettingsOpener {
    /// Set when the app starts.
    static var openSettings: (() -> Void)?

    static func open() {
        NSApp.activate()
        openSettings?()
    }
}
