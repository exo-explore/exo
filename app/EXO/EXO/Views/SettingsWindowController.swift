import AppKit
import os.log

/// Opens the Settings window, the app's SwiftUI `Settings` scene, in front
/// and active (see `WindowActivation`).
///
/// SwiftUI only opens the scene through `openSettings`, which can be read only
/// inside a SwiftUI view, or through the "Settings…" item it puts in the app
/// menu. EXO's menu bar menu is AppKit and the app menu isn't shown, so this
/// chooses that app menu item.
@MainActor
final class SettingsWindowController: ObservableObject {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "Settings")

    func open() {
        if !Self.chooseSettingsInAppMenu() {
            Self.logger.error("Couldn't find Settings… in the app menu")
        }
        // SwiftUI creates the window during this turn of the run loop.
        DispatchQueue.main.async {
            Self.bringSettingsToFront()
        }
    }

    /// Chooses SwiftUI's "Settings…" (⌘,) in the app menu, which opens the
    /// `Settings` scene. Says whether the item was found.
    private static func chooseSettingsInAppMenu() -> Bool {
        guard let appMenu = NSApp.mainMenu?.items.first?.submenu,
            let index = appMenu.items.firstIndex(where: {
                $0.keyEquivalent == "," && $0.keyEquivalentModifierMask == .command
            })
        else { return false }
        appMenu.performActionForItem(at: index)
        return true
    }

    static var settingsWindow: NSWindow? {
        NSApp.windows.first { $0.identifier?.rawValue.contains("Settings") == true }
    }

    /// Brings Settings in front, also after a password prompt. When that
    /// prompt closes, macOS hands the focus back to the app that was active
    /// before EXO a moment later, so this is done again once it has.
    static func bringSettingsToFront() {
        func bringForward() {
            guard let window = settingsWindow, window.isVisible else { return }
            WindowActivation.bringToFront(window)
        }
        bringForward()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { bringForward() }
    }
}
