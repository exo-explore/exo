import Foundation

/// What EXO does on its own when the app starts, and whether this is its
/// first run on this Mac. Starting at login is always silent.
struct StartupPreferences {
    static let openDashboardOnStartupKey = "EXOOpenDashboardOnStartup"
    static let openDashboardOnStartupDefault = true
    static let launchAtLoginDefaultKey = "EXOLaunchAtLoginDefault"
    /// Set once the first-run welcome (or, in earlier versions, the welcome
    /// popout) has been seen, so its presence means EXO has run on this Mac
    /// before.
    static let onboardingCompletedKey = "EXOOnboardingCompleted"
    /// Whether the welcome window is still due; see `FirstRun`.
    static let firstRunKey = "EXOFirstRun"

    /// Keys that only an EXO that has already run on this Mac writes: the
    /// welcome popout (1.0.x), the launch-at-login default, the Local Network
    /// check, and Sparkle's first launch.
    static let hasRunBeforeKeys = [
        onboardingCompletedKey, launchAtLoginDefaultKey,
        "LocalNetworkChecker.hasCompletedInitialCheck", "SUHasLaunchedBefore",
    ]

    /// Whether the app should register itself as a login item when it starts.
    enum LaunchAtLoginDefault: String {
        /// The user ticked "Open EXO at login" in the welcome window, and
        /// registering hasn't succeeded yet (e.g. the app was running from
        /// the disk image).
        case pending
        /// Registered, or left to the user. The app no longer changes the
        /// login item on its own.
        case applied
    }

    /// Whether the welcome window is still due.
    enum FirstRun: String {
        case welcome
        case done
    }

    let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    // MARK: - First run

    /// Decides, as the app starts and before anything else writes defaults,
    /// whether to show the welcome window. A new install shows it until it
    /// has been seen; an upgrade from a version that already ran never does.
    func welcomeIsDue() -> Bool {
        let state = Self.firstRun(
            stored: defaults.string(forKey: Self.firstRunKey).flatMap(FirstRun.init(rawValue:)),
            hasRunBefore: Self.hasRunBeforeKeys.contains { defaults.object(forKey: $0) != nil })
        defaults.set(state.rawValue, forKey: Self.firstRunKey)
        if state == .done, !onboardingCompleted {
            // An upgrade: as if the welcome had been seen.
            markOnboardingCompleted()
        }
        return state == .welcome
    }

    static func firstRun(stored: FirstRun?, hasRunBefore: Bool) -> FirstRun {
        stored ?? (hasRunBefore ? .done : .welcome)
    }

    /// The welcome window was seen: finished, or closed early.
    func markWelcomeSeen() {
        defaults.set(FirstRun.done.rawValue, forKey: Self.firstRunKey)
        markOnboardingCompleted()
    }

    // MARK: - Dashboard

    /// Whether to open the web dashboard in the browser when exo first comes
    /// up after the user opens the app (never at login). Settings writes this
    /// key directly via `@AppStorage`.
    var openDashboardOnStartup: Bool {
        defaults.object(forKey: Self.openDashboardOnStartupKey) as? Bool
            ?? Self.openDashboardOnStartupDefault
    }

    /// Whether the first-run welcome has been seen.
    var onboardingCompleted: Bool {
        defaults.bool(forKey: Self.onboardingCompletedKey)
    }

    func markOnboardingCompleted() {
        defaults.set(true, forKey: Self.onboardingCompletedKey)
    }

    enum StartupDashboard: Equatable {
        case none
        case openDashboard
    }

    /// Whether to open the dashboard when exo first comes up after the app
    /// starts: only when the user opened EXO by hand, after the first run,
    /// and while the setting is on. A login item that opens a browser tab at
    /// every login is an interruption, and the first run ends with the
    /// welcome window's own Open Dashboard button instead.
    func startupDashboard(launch: AppLaunchKind) -> StartupDashboard {
        guard launch == .manual, openDashboardOnStartup, onboardingCompleted else { return .none }
        return .openDashboard
    }

    // MARK: - Launch at login

    /// Records the welcome window's "Open EXO at login" choice. A ticked box
    /// is registered as the app starts (now, or at the next launch from a
    /// stable location); an unticked one is left alone from then on.
    func chooseLaunchAtLogin(_ open: Bool) {
        defaults.set(
            (open ? LaunchAtLoginDefault.pending : .applied).rawValue,
            forKey: Self.launchAtLoginDefaultKey)
    }

    /// Whether to register the app as a login item now: only when the user
    /// asked for it in the welcome window and it hasn't been registered yet.
    /// Registering is skipped while the app runs translocated (opened
    /// straight from Downloads or the disk image), where it would register a
    /// temporary copy. Nothing is registered without that choice: an upgrade
    /// keeps whatever login item it had.
    func shouldRegisterLaunchAtLoginOnStartup(isTranslocated: Bool) -> Bool {
        let stored = defaults.string(forKey: Self.launchAtLoginDefaultKey)
            .flatMap(LaunchAtLoginDefault.init(rawValue:))
        return stored == .pending && !isTranslocated
    }

    /// Records that the app should no longer change the login item on its
    /// own: registering succeeded, or the user changed the setting.
    func markLaunchAtLoginDefaultApplied() {
        defaults.set(LaunchAtLoginDefault.applied.rawValue, forKey: Self.launchAtLoginDefaultKey)
    }

    /// Forgets the startup choices so that reinstalling after the in-app
    /// Uninstall behaves like a new install.
    func resetForUninstall() {
        for key in Self.hasRunBeforeKeys + [Self.openDashboardOnStartupKey, Self.firstRunKey] {
            defaults.removeObject(forKey: key)
        }
    }

    /// Whether `bundlePath` is an App Translocation path, used when a
    /// quarantined app is opened from Downloads or a disk image.
    static func isTranslocated(bundlePath: String) -> Bool {
        bundlePath.contains("/AppTranslocation/")
    }
}
