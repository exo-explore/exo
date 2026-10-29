//
//  EXOApp.swift
//  EXO
//
//  Created by Sami Khan on 2025-11-22.
//

import AppKit
import ServiceManagement
import SwiftUI
import os.log

struct EXOApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var controller: ExoProcessController
    @StateObject private var stateService: ClusterStateService
    @StateObject private var networkStatusService: NetworkStatusService
    @StateObject private var localNetworkChecker: LocalNetworkChecker
    @StateObject private var updater: SparkleUpdater
    @StateObject private var thunderboltBridgeService: ThunderboltBridgeService
    @StateObject private var settingsWindowController: SettingsWindowController
    @StateObject private var bugReportWindowController: BugReportWindowController
    @StateObject private var menuBar: MenuBarController

    init() {
        // As the unit tests' host app, start nothing: no welcome, no login
        // item, no exo.
        let isHostingTests =
            ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
        // Decided first, before anything else writes this app's defaults.
        let isFirstRun = !isHostingTests && StartupPreferences().welcomeIsDue()

        let controller = ExoProcessController()
        let updater = SparkleUpdater(processController: controller)
        AppDelegate.stopExoBeforeQuitting = { await controller.stopAndWaitForExit() }
        _controller = StateObject(wrappedValue: controller)
        let service = ClusterStateService()
        service.resetWhenExoStarts(controller.$status)
        controller.modelsRunningOnThisMac = { [weak service] in service?.isRunningModelsOnThisMac }
        _stateService = StateObject(wrappedValue: service)
        let networkStatus = NetworkStatusService()
        _networkStatusService = StateObject(wrappedValue: networkStatus)
        let localNetwork = LocalNetworkChecker()
        _localNetworkChecker = StateObject(wrappedValue: localNetwork)
        _updater = StateObject(wrappedValue: updater)
        let thunderboltBridge = ThunderboltBridgeService(clusterStateService: service)
        _thunderboltBridgeService = StateObject(wrappedValue: thunderboltBridge)
        let settingsWindow = SettingsWindowController()
        _settingsWindowController = StateObject(wrappedValue: settingsWindow)
        let bugReportWindow = BugReportWindowController()
        _bugReportWindowController = StateObject(wrappedValue: bugReportWindow)
        let menuBar = MenuBarController(
            controller: controller, stateService: service, localNetworkChecker: localNetwork,
            thunderboltBridgeService: thunderboltBridge, updater: updater,
            bugReportWindowController: bugReportWindow,
            startupDashboardOpener: StartupDashboardOpener(isFirstRun: isFirstRun))
        _menuBar = StateObject(wrappedValue: menuBar)
        SettingsOpener.openSettings = { settingsWindow.open() }
        guard !isHostingTests else { return }

        menuBar.install()
        LaunchAtLoginHelper.registerIfChosen()
        // Nothing asks for an administrator password at launch: Thunderbolt
        // networking (NetworkSetupHelper) is offered when it matters.
        service.startPolling()
        networkStatus.startPolling()
        // exo, and the Local Network check (which, like exo, makes macOS ask
        // for Local Network access), start straight away, except on the first
        // run, where the welcome window explains the prompt first.
        let startExo = {
            localNetwork.startPeriodicChecking(interval: isFirstRun ? 3 : 10)
            controller.launchWhenPreviousExoHasExited()
        }
        guard isFirstRun else {
            startExo()
            AppDelegate.onReopen = { menuBar.handleReopen() }
            return
        }
        let welcome = WelcomeWindowController(
            controller: controller, localNetworkChecker: localNetwork, startExo: startExo)
        AppDelegate.welcomeWindow = welcome
        AppDelegate.launchAsRegularApp = true
        AppDelegate.onFinishLaunching = { welcome.show() }
        AppDelegate.onReopen = {
            if welcome.isShowing {
                welcome.bringToFront()
            } else {
                menuBar.handleReopen()
            }
        }
    }

    /// The menu bar item is AppKit (`MenuBarController`); the one SwiftUI
    /// scene is the Settings window.
    var body: some Scene {
        Settings {
            SettingsView()
                .environmentObject(controller)
                .environmentObject(updater)
                .environmentObject(stateService)
                .environmentObject(networkStatusService)
                .environmentObject(localNetworkChecker)
        }
    }
}

/// Helper for managing EXO's launch-at-login registration
enum LaunchAtLoginHelper {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "LaunchAtLogin")

    static var status: SMAppService.Status {
        SMAppService.mainApp.status
    }

    /// Registers or unregisters EXO as a login item, throwing macOS's error if
    /// it refuses.
    static func setEnabled(_ enabled: Bool) throws {
        let status = SMAppService.mainApp.status
        if enabled {
            guard status != .enabled else { return }
            try SMAppService.mainApp.register()
            logger.info("Registered EXO for launch at login")
        } else {
            guard status == .enabled || status == .requiresApproval else { return }
            try SMAppService.mainApp.unregister()
            logger.info("Unregistered EXO from launch at login")
        }
    }

    /// Registers EXO to launch at login. Returns whether it is registered.
    @discardableResult
    static func enable() -> Bool {
        do {
            try setEnabled(true)
            return true
        } catch {
            logger.error(
                "Failed to register EXO for launch at login: \(error.localizedDescription, privacy: .public)"
            )
            return false
        }
    }

    /// Unregisters EXO from launching at login
    static func disable() {
        do {
            try setEnabled(false)
        } catch {
            logger.error(
                "Failed to unregister EXO from launch at login: \(error.localizedDescription, privacy: .public)"
            )
        }
    }

    /// Registers EXO as a login item if the user ticked "Open EXO at login"
    /// in the welcome window and it isn't registered yet. Waits for a launch
    /// from a stable location if the app runs translocated.
    static func registerIfChosen(preferences: StartupPreferences = StartupPreferences()) {
        let isTranslocated = StartupPreferences.isTranslocated(bundlePath: Bundle.main.bundlePath)
        guard preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: isTranslocated)
        else { return }
        if enable() {
            preferences.markLaunchAtLoginDefaultApplied()
        }
    }
}
