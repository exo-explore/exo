import Foundation

/// Opens the web dashboard the first time exo is up after the app starts:
/// with the welcome popout until that has been seen once, then directly.
/// Switching exo off and on, or Save & Restart in Settings, doesn't open it
/// again, and nothing opens while "Open dashboard on startup" is off.
@MainActor
final class StartupDashboardOpener {
    private let preferences: StartupPreferences
    private let showWelcome: (_ onSeen: @escaping () -> Void) -> Void
    private let openDashboard: () -> Void
    private var hasHandledFirstStart = false

    init(
        preferences: StartupPreferences,
        showWelcome: @escaping (_ onSeen: @escaping () -> Void) -> Void,
        openDashboard: @escaping () -> Void
    ) {
        self.preferences = preferences
        self.showWelcome = showWelcome
        self.openDashboard = openDashboard
    }

    convenience init() {
        let popout = FirstLaunchPopout()
        self.init(
            preferences: StartupPreferences(),
            showWelcome: { onSeen in
                popout.onComplete = onSeen
                popout.show()
            },
            openDashboard: { FirstLaunchPopout.openDashboard() }
        )
    }

    /// Called when exo's API answers. Only the first call has an effect.
    func exoIsUp() {
        guard !hasHandledFirstStart else { return }
        hasHandledFirstStart = true
        switch preferences.startupDashboard {
        case .none:
            break
        case .showWelcome:
            let preferences = preferences
            showWelcome { preferences.markOnboardingCompleted() }
        case .openDashboard:
            openDashboard()
        }
    }
}
