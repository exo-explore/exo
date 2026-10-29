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

    func applicationDidFinishLaunching(_ notification: Notification) {
        Self.launchKind = AppLaunchKind.ofCurrentAppleEvent()
        Self.logger.notice(
            "EXO was opened \(Self.launchKind == .atLogin ? "at login" : "manually", privacy: .public)"
        )
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
