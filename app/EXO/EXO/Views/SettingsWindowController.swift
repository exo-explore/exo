import AppKit
import SwiftUI

/// Manages a standalone native macOS Settings window.
/// Ensures only one instance exists and brings it to front on repeated opens.
@MainActor
final class SettingsWindowController: ObservableObject {
    private var window: NSWindow?

    func open(
        controller: ExoProcessController,
        updater: SparkleUpdater,
        networkStatusService: NetworkStatusService,
        thunderboltBridgeService: ThunderboltBridgeService,
        stateService: ClusterStateService
    ) {
        if let existing = window, existing.isVisible {
            existing.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let settingsView = SettingsView()
            .environmentObject(controller)
            .environmentObject(updater)
            .environmentObject(networkStatusService)
            .environmentObject(thunderboltBridgeService)
            .environmentObject(stateService)

        let hostingView = NSHostingView(rootView: settingsView)

        let newWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 450, height: 400),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        newWindow.title = "EXO Settings"
        newWindow.contentView = hostingView
        newWindow.center()
        newWindow.isReleasedWhenClosed = false
        newWindow.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        window = newWindow
    }
}
