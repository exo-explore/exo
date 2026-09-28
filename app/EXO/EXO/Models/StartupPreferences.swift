import Foundation

/// What EXO does on its own when the app starts. Both behaviours default to
/// on, as they were before they became settings, and are changed from
/// Settings → General → Startup.
struct StartupPreferences {
    static let openDashboardOnStartupKey = "EXOOpenDashboardOnStartup"
    static let openDashboardOnStartupDefault = true
    static let launchAtLoginDefaultAppliedKey = "EXOLaunchAtLoginDefaultApplied"
    /// Set once the welcome popout has been dismissed, so its presence means
    /// EXO has run on this Mac before.
    static let onboardingCompletedKey = "EXOOnboardingCompleted"

    let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    /// Whether to show the welcome popout, which opens the web dashboard in
    /// the browser, when exo starts.
    var openDashboardOnStartup: Bool {
        get {
            defaults.object(forKey: Self.openDashboardOnStartupKey) as? Bool
                ?? Self.openDashboardOnStartupDefault
        }
        nonmutating set {
            defaults.set(newValue, forKey: Self.openDashboardOnStartupKey)
        }
    }

    /// Whether the app should register itself as a login item as it starts.
    ///
    /// Only a new install is opted in, once. After that the login item is
    /// changed only from Settings, so turning it off there or removing EXO in
    /// System Settings → General → Login Items sticks across launches.
    var shouldRegisterLaunchAtLoginOnStartup: Bool {
        Self.shouldRegisterLaunchAtLogin(
            defaultApplied: defaults.bool(forKey: Self.launchAtLoginDefaultAppliedKey),
            hasRunBefore: defaults.object(forKey: Self.onboardingCompletedKey) != nil
        )
    }

    /// Records that the launch-at-login default has been applied, so later
    /// launches leave the login item alone.
    func markLaunchAtLoginDefaultApplied() {
        defaults.set(true, forKey: Self.launchAtLoginDefaultAppliedKey)
    }

    /// `hasRunBefore` covers upgrades from versions that registered the login
    /// item on every launch: if such a user has since removed it, it stays
    /// removed.
    static func shouldRegisterLaunchAtLogin(defaultApplied: Bool, hasRunBefore: Bool) -> Bool {
        !defaultApplied && !hasRunBefore
    }
}
