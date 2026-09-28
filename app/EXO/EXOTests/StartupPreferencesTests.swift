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
}
