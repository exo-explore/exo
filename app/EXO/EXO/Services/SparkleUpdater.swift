import AppKit
import Sparkle
import UserNotifications
import os.log

/// Sparkle, set up for an app that lives in the menu bar. Sparkle checks on
/// its own schedule (SUScheduledCheckInterval, at least an hour) and never
/// opens its update window by itself: an update it finds is shown gently, as
/// `availableUpdateVersion` for the menu bar and one quiet notification per
/// version. Sparkle's window opens when the user asks for it.
@MainActor
final class SparkleUpdater: NSObject, ObservableObject {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "Updates")

    /// The version of an update Sparkle found on a scheduled check that the
    /// user hasn't installed, skipped or put off yet.
    @Published private(set) var availableUpdateVersion: String?

    private let controller: SPUStandardUpdaterController
    private let updaterDelegate: ExoUpdaterDelegate
    private let userDriverDelegate: ExoUserDriverDelegate
    private let notifier: UpdateNotifier

    init(processController: ExoProcessController) {
        let updaterDelegate = ExoUpdaterDelegate(processController: processController)
        let userDriverDelegate = ExoUserDriverDelegate()
        let notifier = UpdateNotifier()
        self.updaterDelegate = updaterDelegate
        self.userDriverDelegate = userDriverDelegate
        self.notifier = notifier
        controller = SPUStandardUpdaterController(
            startingUpdater: true,
            updaterDelegate: updaterDelegate,
            userDriverDelegate: userDriverDelegate
        )
        super.init()

        userDriverDelegate.onScheduledUpdateFound = { [weak self] update in
            let version = update.displayVersionString
            Self.logger.info("Update \(version, privacy: .public) is available")
            self?.availableUpdateVersion = version
            self?.notifier.notifyOnce(aboutVersion: version)
        }
        userDriverDelegate.onUserAttention = { [weak self] in
            self?.notifier.removeDeliveredNotifications()
        }
        userDriverDelegate.onSessionFinished = { [weak self] in
            self?.availableUpdateVersion = nil
        }
        notifier.onOpen = { [weak self] in
            self?.checkForUpdates()
        }
        UNUserNotificationCenter.current().delegate = notifier
    }

    /// Opens Sparkle's window: the update it already found, or a new check.
    func checkForUpdates() {
        NSApp.activate()
        controller.checkForUpdates(nil)
    }

    /// The updater, for settings such as checking and downloading
    /// automatically.
    var updater: SPUUpdater {
        controller.updater
    }

    /// Settings › General › "Check for updates automatically". On unless the
    /// user turned it off (SUEnableAutomaticChecks in Info.plist).
    var automaticallyChecksForUpdates: Bool {
        get { controller.updater.automaticallyChecksForUpdates }
        set {
            objectWillChange.send()
            controller.updater.automaticallyChecksForUpdates = newValue
        }
    }

    /// Settings › General › "Download and install updates automatically".
    /// Off unless the user turns it on.
    var automaticallyDownloadsUpdates: Bool {
        get { controller.updater.automaticallyDownloadsUpdates }
        set {
            objectWillChange.send()
            controller.updater.automaticallyDownloadsUpdates = newValue
        }
    }
}

private final class ExoUpdaterDelegate: NSObject, SPUUpdaterDelegate {
    private weak var processController: ExoProcessController?

    init(processController: ExoProcessController) {
        self.processController = processController
    }

    nonisolated func updaterWillRelaunchApplication(_ updater: SPUUpdater) {
        Task { @MainActor in
            guard let controller = self.processController else { return }
            controller.cancelPendingLaunch()
            controller.stop()
        }
    }
}

/// Sparkle's gentle reminders for background apps: Sparkle tells the app
/// about an update it found instead of putting a window up behind other apps
/// (or at login). See https://sparkle-project.org/documentation/gentle-reminders/
private final class ExoUserDriverDelegate: NSObject, SPUStandardUserDriverDelegate {
    var onScheduledUpdateFound: ((SUAppcastItem) -> Void)?
    var onUserAttention: (() -> Void)?
    var onSessionFinished: (() -> Void)?

    var supportsGentleScheduledUpdateReminders: Bool {
        true
    }

    /// Never let a scheduled check open Sparkle's window: right after launch
    /// that would be at login, and later it would open behind other apps.
    func standardUserDriverShouldHandleShowingScheduledUpdate(
        _ update: SUAppcastItem, andInImmediateFocus immediateFocus: Bool
    ) -> Bool {
        false
    }

    func standardUserDriverWillHandleShowingUpdate(
        _ handleShowingUpdate: Bool, forUpdate update: SUAppcastItem, state: SPUUserUpdateState
    ) {
        // "Check for Updates…" shows its own window; only scheduled finds
        // need a reminder.
        guard !state.userInitiated else { return }
        onScheduledUpdateFound?(update)
    }

    func standardUserDriverDidReceiveUserAttention(forUpdate update: SUAppcastItem) {
        onUserAttention?()
    }

    func standardUserDriverWillFinishUpdateSession() {
        onSessionFinished?()
    }
}

/// Posts one quiet notification per update version, without asking for
/// permission: provisional authorization delivers to Notification Center
/// only. Clicking it shows the update.
final class UpdateNotifier: NSObject, UNUserNotificationCenterDelegate {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "Updates")
    static let lastNotifiedVersionKey = "EXOLastNotifiedUpdateVersion"
    private static let identifierPrefix = "exo-update-"

    var onOpen: (() -> Void)?
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    /// The same identifier for every post about a version, so a repeat
    /// replaces the earlier one instead of stacking up.
    static func identifier(forVersion version: String) -> String {
        identifierPrefix + version
    }

    /// Whether a notification about `version` is still due: Sparkle reports
    /// a pending update at every scheduled check, but people are told once.
    static func shouldNotify(aboutVersion version: String, lastNotifiedVersion: String?) -> Bool {
        version != lastNotifiedVersion
    }

    func notifyOnce(aboutVersion version: String) {
        guard
            Self.shouldNotify(
                aboutVersion: version,
                lastNotifiedVersion: defaults.string(forKey: Self.lastNotifiedVersionKey))
        else { return }
        defaults.set(version, forKey: Self.lastNotifiedVersionKey)

        let center = UNUserNotificationCenter.current()
        center.requestAuthorization(options: [.alert, .provisional]) { granted, error in
            guard granted else {
                Self.logger.notice(
                    "Not posting the update notification: notifications are off for EXO (\(error?.localizedDescription ?? "denied", privacy: .public))"
                )
                return
            }
            let content = UNMutableNotificationContent()
            content.title = "Update Available"
            content.body = "Version \(version) is ready to install."
            content.interruptionLevel = .passive
            center.add(
                UNNotificationRequest(
                    identifier: Self.identifier(forVersion: version), content: content,
                    trigger: nil)
            ) { error in
                if let error {
                    Self.logger.error(
                        "Couldn't post the update notification: \(error.localizedDescription, privacy: .public)"
                    )
                }
            }
        }
    }

    func removeDeliveredNotifications() {
        let center = UNUserNotificationCenter.current()
        center.getDeliveredNotifications { notifications in
            let identifiers = notifications.map(\.request.identifier)
                .filter { $0.hasPrefix(Self.identifierPrefix) }
            center.removeDeliveredNotifications(withIdentifiers: identifiers)
        }
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) ->
            Void
    ) {
        completionHandler([.banner, .list])
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        if response.notification.request.identifier.hasPrefix(Self.identifierPrefix) {
            DispatchQueue.main.async { [onOpen] in
                onOpen?()
            }
        }
        completionHandler()
    }
}
