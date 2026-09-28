import Foundation

/// What EXO does on its own when the app starts. Both behaviours default to
/// on, as they were before they became settings, and are changed from
/// Settings → General → Startup.
struct StartupPreferences {
    static let openDashboardOnStartupKey = "EXOOpenDashboardOnStartup"
    static let openDashboardOnStartupDefault = true
    static let launchAtLoginDefaultKey = "EXOLaunchAtLoginDefault"
    /// Set once the welcome popout has been dismissed, so its presence means
    /// EXO has run on this Mac before.
    static let onboardingCompletedKey = "EXOOnboardingCompleted"

    /// Whether the new-install default of launching at login still has to be
    /// applied.
    enum LaunchAtLoginDefault: String {
        /// A new install that hasn't been registered as a login item yet.
        case pending
        /// Registered, or left to the user (an upgrade, or the user has
        /// changed the setting). The app no longer changes the login item on
        /// its own.
        case applied
    }

    let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    /// Whether to show the welcome popout, which opens the web dashboard in
    /// the browser, when exo starts. Settings writes this key directly via
    /// `@AppStorage`.
    var openDashboardOnStartup: Bool {
        defaults.object(forKey: Self.openDashboardOnStartupKey) as? Bool
            ?? Self.openDashboardOnStartupDefault
    }

    /// Decides, as the app starts, whether to register it as a login item now.
    ///
    /// Only a new install is opted in: it stays `pending` until registering
    /// succeeds, and registering is skipped while the app runs translocated
    /// (opened straight from Downloads or the disk image), where it would
    /// register a temporary copy. An upgrade from a version that registered on
    /// every launch is left as it is, so a login item the user removed stays
    /// removed. Once applied, only Settings changes the login item.
    func shouldRegisterLaunchAtLoginOnStartup(isTranslocated: Bool) -> Bool {
        let state = Self.launchAtLoginDefault(
            stored: defaults.string(forKey: Self.launchAtLoginDefaultKey)
                .flatMap(LaunchAtLoginDefault.init(rawValue:)),
            hasRunBefore: defaults.object(forKey: Self.onboardingCompletedKey) != nil
        )
        defaults.set(state.rawValue, forKey: Self.launchAtLoginDefaultKey)
        return state == .pending && !isTranslocated
    }

    /// Records that the app should no longer change the login item on its
    /// own: registering succeeded, or the user changed the setting.
    func markLaunchAtLoginDefaultApplied() {
        defaults.set(LaunchAtLoginDefault.applied.rawValue, forKey: Self.launchAtLoginDefaultKey)
    }

    /// Forgets the startup choices so that reinstalling after the in-app
    /// Uninstall behaves like a new install.
    func resetForUninstall() {
        for key in [
            Self.launchAtLoginDefaultKey, Self.onboardingCompletedKey,
            Self.openDashboardOnStartupKey,
        ] {
            defaults.removeObject(forKey: key)
        }
    }

    /// The first launch of a version with this setting classifies the Mac:
    /// if EXO has run here before it is an upgrade (`applied`), otherwise a
    /// new install (`pending`). After that the stored value is used.
    static func launchAtLoginDefault(
        stored: LaunchAtLoginDefault?, hasRunBefore: Bool
    ) -> LaunchAtLoginDefault {
        stored ?? (hasRunBefore ? .applied : .pending)
    }

    /// Whether `bundlePath` is an App Translocation path, used when a
    /// quarantined app is opened from Downloads or a disk image.
    static func isTranslocated(bundlePath: String) -> Bool {
        bundlePath.contains("/AppTranslocation/")
    }
}
