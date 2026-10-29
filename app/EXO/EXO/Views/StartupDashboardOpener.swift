import AppKit

/// Opens the web dashboard the first time exo is up after the user opened
/// the app, while "Open dashboard on startup" is on. Nothing opens when the
/// app was started at login, or on the first run, which ends with the
/// welcome window's own Open Dashboard button. Switching exo off and on, or
/// restarting it, doesn't open it again.
@MainActor
final class StartupDashboardOpener {
    private let preferences: StartupPreferences
    private let launchKind: () -> AppLaunchKind
    private let isFirstRun: Bool
    private let openDashboard: () -> Void
    private var hasHandledFirstStart = false

    /// `launchKind` is read when exo comes up, after the app has finished
    /// launching and knows how it was opened.
    init(
        preferences: StartupPreferences,
        launchKind: @escaping () -> AppLaunchKind,
        isFirstRun: Bool,
        openDashboard: @escaping () -> Void
    ) {
        self.preferences = preferences
        self.launchKind = launchKind
        self.isFirstRun = isFirstRun
        self.openDashboard = openDashboard
    }

    convenience init(isFirstRun: Bool) {
        self.init(
            preferences: StartupPreferences(),
            launchKind: { AppDelegate.launchKind },
            isFirstRun: isFirstRun,
            openDashboard: {
                guard let url = URL(string: MenuBarMenu.dashboardURL) else { return }
                NSWorkspace.shared.open(url)
            }
        )
    }

    /// Called when exo's API answers. Only the first call has an effect.
    func exoIsUp() {
        guard !hasHandledFirstStart else { return }
        hasHandledFirstStart = true
        guard !isFirstRun else { return }
        if preferences.startupDashboard(launch: launchKind()) == .openDashboard {
            openDashboard()
        }
    }
}
