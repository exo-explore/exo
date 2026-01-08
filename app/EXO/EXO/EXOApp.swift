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
import Sparkle
import SwiftUI
import UserNotifications
import os.log

@main
struct EXOApp: App {
    @StateObject private var controller: ExoProcessController
    @StateObject private var stateService: ClusterStateService
    @StateObject private var networkStatusService: NetworkStatusService
    @StateObject private var localNetworkChecker: LocalNetworkChecker
    @StateObject private var updater: SparkleUpdater
    private let terminationObserver: TerminationObserver
    private let ciContext = CIContext(options: nil)

    init() {
        let controller = ExoProcessController()
        let updater = SparkleUpdater(processController: controller)
        terminationObserver = TerminationObserver {
            Task { @MainActor in
                controller.cancelPendingLaunch()
                controller.stop()
            }
        }
        _controller = StateObject(wrappedValue: controller)
        let service = ClusterStateService()
        _stateService = StateObject(wrappedValue: service)
        let networkStatus = NetworkStatusService()
        _networkStatusService = StateObject(wrappedValue: networkStatus)
        let localNetwork = LocalNetworkChecker()
        _localNetworkChecker = StateObject(wrappedValue: localNetwork)
        _updater = StateObject(wrappedValue: updater)
        enableLaunchAtLoginIfNeeded()
        NetworkSetupHelper.ensureLaunchDaemonInstalled()
        // Check local network access BEFORE launching exo
        localNetwork.check()
        controller.scheduleLaunch(after: 15)
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
        } label: {
            menuBarIcon
        }
        .menuBarExtraStyle(.window)
    }

    private var menuBarIcon: some View {
        let baseImage = resizedMenuBarIcon(named: "menubar-icon", size: 26)
        let iconImage: NSImage
        if controller.status == .stopped, let grey = greyscale(image: baseImage) {
            iconImage = grey
        } else {
            iconImage = baseImage ?? NSImage(named: "menubar-icon") ?? NSImage()
        }
        return Image(nsImage: iconImage)
            .accessibilityLabel("EXO")
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

    private func enableLaunchAtLoginIfNeeded() {
        guard SMAppService.mainApp.status != .enabled else { return }
        do {
            try SMAppService.mainApp.register()
        } catch {
            Logger().error(
                "Failed to register EXO for launch at login: \(error.localizedDescription)")
        }
    }
}

final class SparkleUpdater: NSObject, ObservableObject {
    private let controller: SPUStandardUpdaterController
    private let delegateProxy: ExoUpdaterDelegate
    private let notificationDelegate = ExoNotificationDelegate()
    private var periodicCheckTask: Task<Void, Never>?

    init(processController: ExoProcessController) {
        let proxy = ExoUpdaterDelegate(processController: processController)
        delegateProxy = proxy
        controller = SPUStandardUpdaterController(
            startingUpdater: true,
            updaterDelegate: proxy,
            userDriverDelegate: nil
        )
        super.init()
        let center = UNUserNotificationCenter.current()
        center.delegate = notificationDelegate
        center.requestAuthorization(options: [.alert, .sound]) { _, _ in }
        controller.updater.automaticallyChecksForUpdates = true
        controller.updater.automaticallyDownloadsUpdates = false
        controller.updater.updateCheckInterval = 900  // 15 minutes
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak controller] in
            controller?.updater.checkForUpdatesInBackground()
        }
        let updater = controller.updater
        let intervalSeconds = max(60.0, controller.updater.updateCheckInterval)
        let intervalNanos = UInt64(intervalSeconds * 1_000_000_000)
        periodicCheckTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: intervalNanos)
                await MainActor.run {
                    updater.checkForUpdatesInBackground()
                }
            }
        }
    }

    deinit {
        periodicCheckTask?.cancel()
    }

    @MainActor
    func checkForUpdates() {
        controller.checkForUpdates(nil)
    }
}

private final class ExoUpdaterDelegate: NSObject, SPUUpdaterDelegate {
    private weak var processController: ExoProcessController?

    init(processController: ExoProcessController) {
        self.processController = processController
    }

    nonisolated func updater(_ updater: SPUUpdater, didFindValidUpdate item: SUAppcastItem) {
        showNotification(
            title: "Update available",
            body: "EXO \(item.displayVersionString) is ready to install."
        )
    }

    nonisolated func updaterWillRelaunchApplication(_ updater: SPUUpdater) {
        Task { @MainActor in
            guard let controller = self.processController else { return }
            controller.cancelPendingLaunch()
            controller.stop()
        }
    }

    private func showNotification(title: String, body: String) {
        let center = UNUserNotificationCenter.current()
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        let request = UNNotificationRequest(
            identifier: "exo-update-\(UUID().uuidString)",
            content: content,
            trigger: nil
        )
        center.add(request, withCompletionHandler: nil)
    }
}

private final class ExoNotificationDelegate: NSObject, UNUserNotificationCenterDelegate {
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) ->
            Void
    ) {
        completionHandler([.banner, .list, .sound])
    }
}

@MainActor
private final class TerminationObserver {
    private var token: NSObjectProtocol?

    init(onTerminate: @escaping () -> Void) {
        token = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { _ in
            onTerminate()
        }
    }

    deinit {
        if let token {
            NotificationCenter.default.removeObserver(token)
        }
    }
}
