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

    // MARK: - Launch at login

    @Test func newInstallIsRegisteredUntilItSucceedsThenLeftAlone() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)

        #expect(preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
        // Registering failed, and the welcome popout was seen meanwhile: the
        // next launch still retries.
        storage.defaults.set(true, forKey: StartupPreferences.onboardingCompletedKey)
        #expect(preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))

        preferences.markLaunchAtLoginDefaultApplied()  // registered
        for _ in 0..<3 {
            #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
        }
    }

    @Test func translocatedLaunchWaitsForTheInstalledCopy() {
        // Opened straight from the disk image: registering now would point
        // the login item at a temporary copy.
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)

        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: true))
        #expect(preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
    }

    @Test func upgradeFromAVersionThatAlwaysRegisteredIsLeftAlone() {
        // Earlier versions registered on every launch and recorded that the
        // welcome popout had been seen, but had no launch-at-login setting. A
        // login item the user removed must stay removed.
        let storage = TemporaryDefaults()
        storage.defaults.set(true, forKey: StartupPreferences.onboardingCompletedKey)
        let preferences = StartupPreferences(defaults: storage.defaults)

        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
        storage.defaults.removeObject(forKey: StartupPreferences.onboardingCompletedKey)
        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
    }

    @Test func changingTheSettingStopsStartupFromRegistering() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: true))

        preferences.markLaunchAtLoginDefaultApplied()  // user turned it off in Settings

        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
    }

    @Test func reinstallAfterInAppUninstallIsANewInstall() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        storage.defaults.set(true, forKey: StartupPreferences.onboardingCompletedKey)
        storage.defaults.set(false, forKey: StartupPreferences.openDashboardOnStartupKey)
        preferences.markLaunchAtLoginDefaultApplied()

        preferences.resetForUninstall()

        #expect(preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: false))
        #expect(preferences.openDashboardOnStartup)
    }

    @Test(
        arguments: [
            (stored: nil, hasRunBefore: false, expected: .pending),
            (stored: nil, hasRunBefore: true, expected: .applied),
            (stored: .pending, hasRunBefore: true, expected: .pending),
            (stored: .applied, hasRunBefore: false, expected: .applied),
        ]
            as [(
                StartupPreferences.LaunchAtLoginDefault?, Bool,
                StartupPreferences.LaunchAtLoginDefault
            )])
    func launchAtLoginDefaultClassification(
        stored: StartupPreferences.LaunchAtLoginDefault?, hasRunBefore: Bool,
        expected: StartupPreferences.LaunchAtLoginDefault
    ) {
        #expect(
            StartupPreferences.launchAtLoginDefault(stored: stored, hasRunBefore: hasRunBefore)
                == expected)
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

    // MARK: - Welcome popout

    @Test func welcomePopoutUntilItHasBeenSeenThenTheDashboardDirectly() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        #expect(!preferences.onboardingCompleted)
        #expect(preferences.startupDashboard(launch: .manual) == .showWelcome)

        preferences.markOnboardingCompleted()

        #expect(preferences.onboardingCompleted)
        #expect(preferences.startupDashboard(launch: .manual) == .openDashboard)
    }

    @Test func upgradeFromAVersionThatShowedThePopoutEveryTimeSkipsIt() {
        // Earlier versions recorded the popout as seen but never read it.
        let storage = TemporaryDefaults()
        storage.defaults.set(true, forKey: StartupPreferences.onboardingCompletedKey)

        #expect(
            StartupPreferences(defaults: storage.defaults).startupDashboard(launch: .manual)
                == .openDashboard)
    }

    @Test func turningOffOpenDashboardOnStartupOpensNothing() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        storage.defaults.set(false, forKey: StartupPreferences.openDashboardOnStartupKey)
        #expect(preferences.startupDashboard(launch: .manual) == .none)

        preferences.markOnboardingCompleted()
        #expect(preferences.startupDashboard(launch: .manual) == .none)
    }

    @Test func reinstallAfterInAppUninstallShowsTheWelcomeAgain() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        preferences.markOnboardingCompleted()

        preferences.resetForUninstall()

        #expect(preferences.startupDashboard(launch: .manual) == .showWelcome)
    }

    // MARK: - Starting at login

    @Test func startingAtLoginOpensNothing() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        #expect(preferences.startupDashboard(launch: .atLogin) == .none)

        preferences.markOnboardingCompleted()

        #expect(preferences.startupDashboard(launch: .atLogin) == .none)
        #expect(preferences.startupDashboard(launch: .manual) == .openDashboard)
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

/// Records what the opener did instead of showing a popover or a browser.
@MainActor
private final class StartupDashboardSpy {
    var welcomesShown = 0
    var dashboardsOpened = 0
    var finishWelcome: (() -> Void)?

    func opener(
        _ preferences: StartupPreferences, launch: AppLaunchKind = .manual
    ) -> StartupDashboardOpener {
        StartupDashboardOpener(
            preferences: preferences,
            launchKind: { launch },
            showWelcome: { onSeen in
                self.welcomesShown += 1
                self.finishWelcome = onSeen
            },
            openDashboard: { self.dashboardsOpened += 1 }
        )
    }
}

@MainActor
struct StartupDashboardOpenerTests {

    @Test func firstLaunchShowsTheWelcomeOnceAndLaterLaunchesOpenTheDashboard() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        let spy = StartupDashboardSpy()

        // First launch: exo comes up, is switched off and on, and restarted.
        let firstLaunch = spy.opener(preferences)
        for _ in 0..<3 {
            firstLaunch.exoIsUp()
        }
        #expect(spy.welcomesShown == 1)
        #expect(spy.dashboardsOpened == 0)
        spy.finishWelcome?()  // dismissed, used, or counted down

        // Every later manual launch: the dashboard, once, no popout.
        for _ in 0..<2 {
            let laterLaunch = spy.opener(preferences)
            laterLaunch.exoIsUp()
            laterLaunch.exoIsUp()
        }
        #expect(spy.welcomesShown == 1)
        #expect(spy.dashboardsOpened == 2)
    }

    @Test func loginLaunchesStaySilent() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        let spy = StartupDashboardSpy()

        // Even before the welcome has been seen.
        spy.opener(preferences, launch: .atLogin).exoIsUp()
        preferences.markOnboardingCompleted()
        spy.opener(preferences, launch: .atLogin).exoIsUp()

        #expect(spy.welcomesShown == 0)
        #expect(spy.dashboardsOpened == 0)
    }

    @Test func welcomeComesBackIfTheAppQuitBeforeItWasSeen() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)
        let spy = StartupDashboardSpy()

        spy.opener(preferences).exoIsUp()
        spy.opener(preferences).exoIsUp()

        #expect(spy.welcomesShown == 2)
        #expect(spy.dashboardsOpened == 0)
    }

    @Test func nothingOpensWhenTheSettingIsOff() {
        let storage = TemporaryDefaults()
        storage.defaults.set(false, forKey: StartupPreferences.openDashboardOnStartupKey)
        let spy = StartupDashboardSpy()

        spy.opener(StartupPreferences(defaults: storage.defaults)).exoIsUp()

        #expect(spy.welcomesShown == 0)
        #expect(spy.dashboardsOpened == 0)
    }
}
