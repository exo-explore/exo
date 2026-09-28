import Foundation
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

    @Test func newInstallIsRegisteredForLaunchAtLoginOnce() {
        let storage = TemporaryDefaults()
        let preferences = StartupPreferences(defaults: storage.defaults)

        #expect(preferences.shouldRegisterLaunchAtLoginOnStartup)
        preferences.markLaunchAtLoginDefaultApplied()
        #expect(!preferences.shouldRegisterLaunchAtLoginOnStartup)
    }

    @Test func loginItemIsNotReRegisteredOnLaterLaunches() {
        // After the first launch the user may have turned launch at login off
        // in Settings or removed EXO in System Settings → Login Items.
        let storage = TemporaryDefaults()
        StartupPreferences(defaults: storage.defaults).markLaunchAtLoginDefaultApplied()

        for _ in 0..<3 {
            #expect(
                !StartupPreferences(defaults: storage.defaults).shouldRegisterLaunchAtLoginOnStartup
            )
        }
    }

    @Test func upgradeFromAVersionThatAlwaysRegisteredDoesNotReRegister() {
        // Earlier versions registered on every launch and recorded that the
        // welcome popout had been seen, but had no launch-at-login setting.
        let storage = TemporaryDefaults()
        storage.defaults.set(true, forKey: StartupPreferences.onboardingCompletedKey)

        #expect(
            !StartupPreferences(defaults: storage.defaults).shouldRegisterLaunchAtLoginOnStartup)
    }

    @Test(arguments: [
        (defaultApplied: false, hasRunBefore: false, expected: true),
        (defaultApplied: false, hasRunBefore: true, expected: false),
        (defaultApplied: true, hasRunBefore: false, expected: false),
        (defaultApplied: true, hasRunBefore: true, expected: false),
    ])
    func registrationDecision(defaultApplied: Bool, hasRunBefore: Bool, expected: Bool) {
        #expect(
            StartupPreferences.shouldRegisterLaunchAtLogin(
                defaultApplied: defaultApplied, hasRunBefore: hasRunBefore) == expected)
    }

    // MARK: - Open dashboard on startup

    @Test func dashboardOpensOnStartupByDefault() {
        let storage = TemporaryDefaults()

        #expect(StartupPreferences(defaults: storage.defaults).openDashboardOnStartup)
    }

    @Test func turningOffOpenDashboardOnStartupPersists() {
        let storage = TemporaryDefaults()
        StartupPreferences(defaults: storage.defaults).openDashboardOnStartup = false

        #expect(!StartupPreferences(defaults: storage.defaults).openDashboardOnStartup)
        #expect(
            storage.defaults.object(forKey: StartupPreferences.openDashboardOnStartupKey) as? Bool
                == false)

        StartupPreferences(defaults: storage.defaults).openDashboardOnStartup = true
        #expect(StartupPreferences(defaults: storage.defaults).openDashboardOnStartup)
    }
}
