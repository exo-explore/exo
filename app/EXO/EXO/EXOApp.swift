//
//  EXOApp.swift
//  EXO
//
//  Created by Sami Khan on 2025-11-22.
//

import AppKit
import CoreImage
import CoreImage.CIFilterBuiltins
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
    private let startupDashboardOpener = StartupDashboardOpener()
    private let ciContext = CIContext(options: nil)

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
        _settingsWindowController = StateObject(wrappedValue: SettingsWindowController())
        _bugReportWindowController = StateObject(wrappedValue: BugReportWindowController())
        // As the unit tests' host app, start nothing: no login item, no exo.
        guard ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil else {
            return
        }
        applyLaunchAtLoginDefault()
        // Nothing asks for an administrator password at launch: Thunderbolt
        // networking (NetworkSetupHelper) is offered when it matters.
        // Check local network access periodically (warning disappears when user grants permission)
        localNetwork.startPeriodicChecking(interval: 10)
        controller.launchWhenPreviousExoHasExited()
        service.startPolling()
        networkStatus.startPolling()
    }

    var body: some Scene {
        MenuBarExtra {
            ContentView()
                .environmentObject(controller)
                .environmentObject(stateService)
                .environmentObject(networkStatusService)
                .environmentObject(localNetworkChecker)
                .environmentObject(updater)
                .environmentObject(thunderboltBridgeService)
                .environmentObject(settingsWindowController)
                .environmentObject(bugReportWindowController)
        } label: {
            menuBarIcon
                // exo's API answering /node_id means it is up and the
                // dashboard can be opened.
                .onReceive(stateService.$localNodeId) { nodeId in
                    if nodeId != nil {
                        startupDashboardOpener.exoIsUp()
                    }
                }
        }
        .menuBarExtraStyle(.window)
    }

    private var menuBarIcon: some View {
        let style = MenuBarIconStyle.showing(controller.status, apiState: stateService.apiState)
        let baseImage = resizedMenuBarIcon(named: "menubar-icon", size: 26)
        var iconImage = baseImage ?? NSImage(named: "menubar-icon") ?? NSImage()
        if style.isGreyscale, let grey = greyscale(image: baseImage) {
            iconImage = grey
        }
        if let badge = style.badge {
            iconImage = badged(iconImage, color: badge.color)
        }
        return Image(nsImage: iconImage)
            .accessibilityLabel(style.accessibilityLabel)
    }

    /// Draws a coloured dot in the bottom-right corner, in the empty space
    /// below the wordmark.
    private func badged(_ image: NSImage, color: NSColor) -> NSImage {
        let size = image.size
        let diameter = (size.width * 0.32).rounded()
        let badgedImage = NSImage(size: size)
        badgedImage.lockFocus()
        defer { badgedImage.unlockFocus() }
        image.draw(in: NSRect(origin: .zero, size: size))
        color.setFill()
        NSBezierPath(
            ovalIn: NSRect(x: size.width - diameter, y: 0, width: diameter, height: diameter)
        ).fill()
        return badgedImage
    }

    private func resizedMenuBarIcon(named: String, size: CGFloat) -> NSImage? {
        guard let original = NSImage(named: named) else {
            print("Failed to load image named: \(named)")
            return nil
        }
        let targetSize = NSSize(width: size, height: size)
        let resized = NSImage(size: targetSize)
        resized.lockFocus()
        defer { resized.unlockFocus() }
        NSGraphicsContext.current?.imageInterpolation = .high
        original.draw(
            in: NSRect(origin: .zero, size: targetSize),
            from: NSRect(origin: .zero, size: original.size),
            operation: .copy,
            fraction: 1.0
        )
        return resized
    }

    private func greyscale(image: NSImage?) -> NSImage? {
        guard
            let image,
            let tiff = image.tiffRepresentation,
            let bitmap = NSBitmapImageRep(data: tiff),
            let cgImage = bitmap.cgImage
        else {
            return nil
        }

        let ciImage = CIImage(cgImage: cgImage)
        let filter = CIFilter.colorControls()
        filter.inputImage = ciImage
        filter.saturation = 0
        filter.brightness = -0.2
        filter.contrast = 0.9

        guard let output = filter.outputImage,
            let rendered = ciContext.createCGImage(output, from: output.extent)
        else {
            return nil
        }

        return NSImage(cgImage: rendered, size: image.size)
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
