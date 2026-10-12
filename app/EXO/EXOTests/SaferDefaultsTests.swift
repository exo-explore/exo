import AppKit
import Testing

@testable import EXO

@MainActor
struct UninstallAlertTests {

    @Test func uninstallKeepsDownloadedModelsUnlessUnticked() {
        let (alert, keepModels) = SettingsView.makeUninstallAlert()

        #expect(keepModels.state == .on)
        #expect(alert.accessoryView === keepModels)
        #expect(alert.buttons.map(\.title) == ["Uninstall", "Cancel"])
        #expect(alert.buttons.first?.hasDestructiveAction == true)
    }
}
