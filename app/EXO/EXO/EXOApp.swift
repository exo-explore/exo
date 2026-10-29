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
        let controller = ExoProcessController()
        let updater = SparkleUpdater(processController: controller)
        AppDelegate.stopExoBeforeQuitting = { await controller.stopAndWaitForExit() }
        _controller = StateObject(wrappedValue: controller)
        let service = ClusterStateService()
        service.resetWhenExoStarts(controller.$status)
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
            bugReportWindowController: bugReportWindow)
        _menuBar = StateObject(wrappedValue: menuBar)
        SettingsOpener.openSettings = {
            settingsWindow.open(
                controller: controller, updater: updater, networkStatusService: networkStatus,
                thunderboltBridgeService: thunderboltBridge, stateService: service)
        }
        AppDelegate.onReopen = { menuBar.handleReopen() }
        // As the unit tests' host app, start nothing: no login item, no exo.
        guard ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil else {
            return
        }
        menuBar.install()
        applyLaunchAtLoginDefault()
        // Nothing asks for an administrator password at launch: Thunderbolt
        // networking (NetworkSetupHelper) is offered when it matters.
        // Check local network access periodically (warning disappears when user grants permission)
        localNetwork.startPeriodicChecking(interval: 10)
        controller.launchWhenPreviousExoHasExited()
        service.startPolling()
        networkStatus.startPolling()
    }

    /// The menu bar item is AppKit (`MenuBarController`); SwiftUI only needs
    /// a scene, and this one shows nothing until Settings moves into it.
    var body: some Scene {
        Settings {
            EmptyView()
        }
    }

    /// Registers a new install as a login item, retrying on later launches
    /// until it succeeds. Upgrades, and launches after that, leave the login
    /// item alone so that turning it off (in Settings or in System Settings →
    /// Login Items) sticks.
    private func applyLaunchAtLoginDefault() {
        let preferences = StartupPreferences()
        let isTranslocated = StartupPreferences.isTranslocated(
            bundlePath: Bundle.main.bundlePath)
        guard preferences.shouldRegisterLaunchAtLoginOnStartup(isTranslocated: isTranslocated)
        else { return }
        if LaunchAtLoginHelper.enable() {
            preferences.markLaunchAtLoginDefaultApplied()
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
}
