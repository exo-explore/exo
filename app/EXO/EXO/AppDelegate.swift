import AppKit
import os.log

/// Receives the application events SwiftUI's `App` doesn't expose.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "Launch")

    /// How this run of the app was opened. Read once launching has finished;
    /// `.manual` until then.
    private(set) static var launchKind: AppLaunchKind = .manual

    /// Stops exo and returns once it has exited. Set by `EXOApp`.
    static var stopExoBeforeQuitting: (@MainActor () async -> Void)?
    /// Shows something when EXO.app is opened again while it runs. Set by
    /// `EXOApp`.
    static var onReopen: (@MainActor () -> Void)?
    /// Runs once the app has finished launching, e.g. to show the welcome
    /// window. Set by `EXOApp`.
    static var onFinishLaunching: (@MainActor () -> Void)?
    /// The first-run welcome window, kept alive while it is shown.
    static var welcomeWindow: WelcomeWindowController?
    /// Whether to launch as a regular app (Dock icon, activated like any app
    /// the user opens) rather than a menu-bar-only one, for the welcome
    /// window. Must be set before launching finishes; macOS doesn't make a
    /// menu-bar-only app active when Finder or Spotlight opens it.
    static var launchAsRegularApp = false

    func applicationWillFinishLaunching(_ notification: Notification) {
        if Self.launchAsRegularApp {
            NSApp.setActivationPolicy(.regular)
        }
    }

    /// Opening EXO.app from Finder or Spotlight while it runs. The menu bar
    /// icon can be hidden (by the notch, or under System Settings › Menu
    /// Bar), so this must show something rather than nothing.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool)
        -> Bool
    {
        Self.onReopen?()
        return false
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        Self.launchKind = AppLaunchKind.ofCurrentAppleEvent()
        Self.logger.notice(
            "EXO was opened \(Self.launchKind == .atLogin ? "at login" : "manually", privacy: .public)"
        )
        Self.onFinishLaunching?()
    }

    /// Quitting EXO (from its menu, with a Quit Apple Event, at logout, or
    /// to relaunch for an update) waits for exo to exit, which takes a second
    /// or two. Otherwise the app is gone before exo has been told to stop,
    /// and an orphaned exo keeps running and holding its PID lock.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let stopExo = Self.stopExoBeforeQuitting else { return .terminateNow }
        Task { @MainActor in
            await stopExo()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}
