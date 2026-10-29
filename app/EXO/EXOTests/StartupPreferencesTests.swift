import AppKit
import Foundation
import SwiftUI
import Testing

@testable import EXO

/// Each test gets its own throwaway defaults. Using an absolute path as the
/// suite name keeps them in a plist inside a temporary directory rather than
/// in ~/Library/Preferences, so nothing touches the app's real preferences.
private final class TemporaryDefaults {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("EXOTests-\(UUID().uuidString)", isDirectory: true)
    let suiteName: String
    let defaults: UserDefaults

    init() {
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        suiteName = directory.appendingPathComponent("defaults").path
        defaults = UserDefaults(suiteName: suiteName)!
    }

    deinit {
        defaults.removePersistentDomain(forName: suiteName)
        try? FileManager.default.removeItem(at: directory)
    }
}

struct StartupPreferencesTests {

    // MARK: - First run

    @Test func newInstallShowsTheWelcomeUntilItHasBeenSeen() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)

        #expect(preferences.welcomeIsDue())
        // Quit before finishing it, then the app ran (writing its usual keys):
        // still due at the next launch.
        storage.defaults.set(true, forKey: "SUHasLaunchedBefore")
        storage.defaults.set(true, forKey: "LocalNetworkChecker.hasCompletedInitialCheck")
        #expect(preferences.welcomeIsDue())

        preferences.markWelcomeSeen()
        #expect(!preferences.welcomeIsDue())
        #expect(preferences.onboardingCompleted)
    }

    @Test(arguments: StartupPreferences.hasRunBeforeKeys)
    func upgradeFromAVersionThatRanNeverSeesTheWelcome(key: String) {
        let storage = TemporaryDefaults()
        storage.defaults.set(true, forKey: key)
        let preferences = StartupPreferences(defaults: storage.defaults)

        #expect(!preferences.welcomeIsDue())
        // Treated as seen, so later manual launches open the dashboard.
        #expect(preferences.onboardingCompleted)
        #expect(preferences.startupDashboard(launch: .manual) == .openDashboard)
    }

    @Test(
        arguments: [
            (stored: nil, hasRunBefore: false, expected: .welcome),
            (stored: nil, hasRunBefore: true, expected: .done),
            (stored: .welcome, hasRunBefore: true, expected: .welcome),
            (stored: .done, hasRunBefore: false, expected: .done),
        ] as [(StartupPreferences.FirstRun?, Bool, StartupPreferences.FirstRun)])
    func firstRunClassification(
        stored: StartupPreferences.FirstRun?, hasRunBefore: Bool,
        expected: StartupPreferences.FirstRun
    ) {
        #expect(StartupPreferences.firstRun(stored: stored, hasRunBefore: hasRunBefore) == expected)
    }

    // MARK: - Launch at login

    @Test func nothingIsRegisteredWithoutTheWelcomesCheckbox() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        // A new install, before the welcome, and an upgrade: no silent
        // registration.
        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
        storage.defaults.set(true, forKey: StartupPreferences.onboardingCompletedKey)
        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
    }

    @Test func aTickedBoxIsRegisteredUntilItSucceedsThenLeftAlone() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        preferences.chooseLaunchAtLogin(true)

        #expect(preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
        // Opened straight from the disk image: registering now would point
        // the login item at a temporary copy, so wait for the installed one.
        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: true))
        #expect(preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))

        preferences.markLaunchAtLoginDefaultApplied()  // registered
        for _ in 0..<3 {
            #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
        }
    }

    @Test func anUntickedBoxIsRespected() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        preferences.chooseLaunchAtLogin(false)

        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
    }

    @Test func reinstallAfterInAppUninstallIsANewInstall() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        storage.defaults.set(true, forKey: "SUHasLaunchedBefore")
        storage.defaults.set(true, forKey: "LocalNetworkChecker.hasCompletedInitialCheck")
        storage.defaults.set(false, forKey: StartupPreferences.openDashboardOnStartupKey)
        #expect(!preferences.welcomeIsDue())
        preferences.chooseLaunchAtLogin(false)

        preferences.resetForUninstall()

        #expect(preferences.welcomeIsDue())
        #expect(preferences.openDashboardOnStartup)
        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
    }

    @Test func detectsTranslocatedBundlePaths() {
        #expect(
            StartupPreferences.isTranslocated(
                bundlePath:
                    "/private/var/folders/xy/abc123/T/AppTranslocation/6F1C2D3E-0000-4000-8000-000000000000/d/EXO.app"
            ))
        #expect(!StartupPreferences.isTranslocated(bundlePath: "/Applications/EXO.app"))
    }

    // MARK: - Open dashboard on startup

    @Test func dashboardOpensOnStartupByDefault() {
        let storage = TemporaryDefaults()

        #expect(StartupPreferences(defaults: storage.defaults).openDashboardOnStartup)
    }

    @Test func settingsToggleControlsOpeningTheDashboard() {
        // The same property wrapper, key and default the Settings toggle uses.
        let storage = TemporaryDefaults()
        let toggle = AppStorage(
            wrappedValue: StartupPreferences.openDashboardOnStartupDefault,
            StartupPreferences.openDashboardOnStartupKey,
            store: storage.defaults
        )
        #expect(toggle.wrappedValue)

        toggle.wrappedValue = false
        #expect(!StartupPreferences(defaults: storage.defaults).openDashboardOnStartup)

        toggle.wrappedValue = true
        #expect(StartupPreferences(defaults: storage.defaults).openDashboardOnStartup)
    }

    @Test func dashboardOpensOnlyOnManualLaunchesAfterTheWelcome() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        #expect(preferences.startupDashboard(launch: .manual) == .none)

        preferences.markWelcomeSeen()

        #expect(preferences.startupDashboard(launch: .manual) == .openDashboard)
        #expect(preferences.startupDashboard(launch: .atLogin) == .none)

        storage.defaults.set(false, forKey: StartupPreferences.openDashboardOnStartupKey)
        #expect(preferences.startupDashboard(launch: .manual) == .none)
    }

    @Test(
        arguments: [
            (
                eventID: AEEventID(kAEOpenApplication), property: OSType(keyAELaunchedAsLogInItem),
                expected: .atLogin
            ),
            (eventID: AEEventID(kAEOpenApplication), property: nil, expected: .manual),
            (
                eventID: AEEventID(kAEOpenApplication),
                property: OSType(keyAELaunchedAsServiceItem), expected: .manual
            ),
            (
                eventID: AEEventID(kAEReopenApplication),
                property: OSType(keyAELaunchedAsLogInItem), expected: .manual
            ),
            (eventID: nil, property: nil, expected: .manual),
        ] as [(AEEventID?, OSType?, AppLaunchKind)])
    func launchKindComesFromTheOpenApplicationEvent(
        eventID: AEEventID?, property: OSType?, expected: AppLaunchKind
    ) {
        #expect(AppLaunchKind.of(eventID: eventID, launchProperty: property) == expected)
    }
}

/// Records what the opener did instead of opening a browser.
@MainActor
private final class StartupDashboardSpy {
    var dashboardsOpened = 0

    func opener(
        _ preferences: StartupPreferences, launch: AppLaunchKind = .manual, firstRun: Bool = false
    ) -> StartupDashboardOpener {
        StartupDashboardOpener(
            preferences: preferences,
            launchKind: { launch },
            isFirstRun: firstRun,
            openDashboard: { self.dashboardsOpened += 1 }
        )
    }
}

@MainActor
struct StartupDashboardOpenerTests {

    @Test func theFirstRunNeverOpensTheBrowserByItself() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        let spy = StartupDashboardSpy()

        let firstRun = spy.opener(preferences, firstRun: true)
        preferences.markWelcomeSeen()  // closed the welcome before exo came up
        firstRun.exoIsUp()

        #expect(spy.dashboardsOpened == 0)
    }

    @Test func laterManualLaunchesOpenTheDashboardOnce() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        preferences.markWelcomeSeen()
        let spy = StartupDashboardSpy()

        for _ in 0..<2 {
            let launch = spy.opener(preferences)
            // exo comes up, is switched off and on, and restarted.
            for _ in 0..<3 {
                launch.exoIsUp()
            }
        }

        #expect(spy.dashboardsOpened == 2)
    }

    @Test func loginLaunchesStaySilent() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        preferences.markWelcomeSeen()
        let spy = StartupDashboardSpy()

        spy.opener(preferences, launch: .atLogin).exoIsUp()

        #expect(spy.dashboardsOpened == 0)
    }

    @Test func nothingOpensWhenTheSettingIsOff() {
        let storage = TemporaryDefaults()
        storage.defaults.set(false, forKey: StartupPreferences.openDashboardOnStartupKey)
        let preferences = StartupPreferences(defaults: storage.defaults)
        preferences.markWelcomeSeen()
        let spy = StartupDashboardSpy()

        spy.opener(preferences).exoIsUp()

        #expect(spy.dashboardsOpened == 0)
    }
}

// MARK: - Welcome window

struct WelcomeFlowTests {

    @Test func continueTwiceAsksForLocalNetworkThenAnAllowLeadsToReady() {
        var flow = WelcomeFlow()
        #expect(flow.pane == .welcome)

        #expect(flow.handle(.continueFromWelcome).isEmpty)
        #expect(flow.pane == .findMacs && flow.localNetwork == .notAsked)

        // Continue starts exo, which makes macOS ask.
        #expect(flow.handle(.askForLocalNetwork) == [.startExo])
        #expect(flow.localNetwork == .asking)
        // Asking twice doesn't start exo twice.
        #expect(flow.handle(.askForLocalNetwork).isEmpty)

        flow.handle(.localNetworkAllowed)
        #expect(flow.pane == .ready && flow.localNetwork == .allowed)
    }

    @Test func aDenialOffersSettingsAndLetsTheUserContinue() {
        var flow = WelcomeFlow()
        flow.handle(.continueFromWelcome)
        flow.handle(.askForLocalNetwork)

        flow.handle(.localNetworkDenied)
        #expect(flow.pane == .findMacs && flow.localNetwork == .denied)

        flow.handle(.continueWithoutLocalNetwork)
        #expect(flow.pane == .ready && flow.localNetwork == .denied)
    }

    @Test func turningAccessOnAfterADenialMovesOn() {
        var flow = WelcomeFlow()
        flow.handle(.continueFromWelcome)
        flow.handle(.askForLocalNetwork)
        flow.handle(.localNetworkDenied)

        flow.handle(.localNetworkAllowed)

        #expect(flow.pane == .ready && flow.localNetwork == .allowed)
    }

    @Test func anUndecidedCheckDoesntHoldTheUserUp() {
        var flow = WelcomeFlow()
        flow.handle(.continueFromWelcome)
        flow.handle(.askForLocalNetwork)

        flow.handle(.localNetworkUndetermined)

        #expect(flow.pane == .ready)
    }

    @Test func resultsBeforeAskingAreIgnored() {
        var flow = WelcomeFlow()
        flow.handle(.localNetworkAllowed)
        flow.handle(.localNetworkDenied)
        #expect(flow.pane == .welcome)
        flow.handle(.continueFromWelcome)
        flow.handle(.localNetworkDenied)
        flow.handle(.continueWithoutLocalNetwork)
        #expect(flow.pane == .findMacs && flow.localNetwork == .notAsked)
    }
}
