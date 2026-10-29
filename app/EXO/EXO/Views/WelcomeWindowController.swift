import AppKit
import Combine
import SwiftUI
import os.log

/// What the welcome window shows and does; see `WelcomeFlow`.
@MainActor
final class WelcomeModel: ObservableObject {
    @Published private(set) var flow = WelcomeFlow()
    @Published var openAtLogin = true
    @Published fileprivate(set) var exoStatus: ExoProcessController.Status = .stopped

    fileprivate var onContinueFromWelcome: () -> Void = {}
    fileprivate var onStartExo: () -> Void = {}
    fileprivate var onOpenDashboard: () -> Void = {}
    fileprivate var onOpenPrivacySettings: () -> Void = {}
    fileprivate var onRestartExo: () -> Void = {}

    func send(_ event: WelcomeFlow.Event) {
        if event == .continueFromWelcome {
            onContinueFromWelcome()
        }
        for effect in flow.handle(event) {
            switch effect {
            case .startExo: onStartExo()
            }
        }
    }

    func openDashboard() { onOpenDashboard() }
    func openPrivacySettings() { onOpenPrivacySettings() }
    func restartExo() { onRestartExo() }
}

/// The first-run welcome: a standard window that replaces the old popout and
/// its countdown. It never opens the browser by itself, and closing it at any
/// point counts as having seen it.
@MainActor
final class WelcomeWindowController: NSObject, NSWindowDelegate {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "Welcome")
    /// How long macOS gets to show its Local Network prompt before a "no"
    /// counts: while the prompt is up, access can look denied.
    private static let localNetworkGracePeriod: TimeInterval = 4

    private let model = WelcomeModel()
    private let controller: ExoProcessController
    private let localNetworkChecker: LocalNetworkChecker
    private let startExo: () -> Void
    private let preferences = StartupPreferences()
    private var window: NSWindow?
    private var cancellables = Set<AnyCancellable>()
    private var askedAt: Date?
    private var exoStarted = false

    var isShowing: Bool { window?.isVisible == true }

    /// `startExo` starts exo and the Local Network check, as on any later
    /// launch.
    init(
        controller: ExoProcessController, localNetworkChecker: LocalNetworkChecker,
        startExo: @escaping () -> Void
    ) {
        self.controller = controller
        self.localNetworkChecker = localNetworkChecker
        self.startExo = startExo
        super.init()
        model.onContinueFromWelcome = { [weak self] in self?.recordLaunchAtLoginChoice() }
        model.onStartExo = { [weak self] in self?.beginExo() }
        model.onOpenDashboard = { [weak self] in
            guard let url = URL(string: MenuBarMenu.dashboardURL) else { return }
            NSWorkspace.shared.open(url)
            self?.window?.close()
        }
        model.onOpenPrivacySettings = {
            NSWorkspace.shared.open(MenuBarController.localNetworkSettingsURL)
        }
        model.onRestartExo = { [weak controller] in
            controller?.stop()
            controller?.launch()
        }
    }

    func show() {
        if let window {
            bringToFront(window)
            return
        }
        let hosting = NSHostingController(rootView: WelcomeView(model: model))
        hosting.sizingOptions = [.preferredContentSize]
        let window = NSWindow(contentViewController: hosting)
        window.styleMask = [.titled, .closable]
        window.title = "Welcome to EXO"
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        self.window = window
        observe()
        bringToFront(window)
    }

    func bringToFront() {
        if let window { bringToFront(window) }
    }

    /// While the welcome is open EXO is a regular app, with a Dock icon and a
    /// place in ⌘-Tab, so the window can't get lost behind others, and macOS
    /// lets it come to the front: a menu-bar-only app opened from Finder or
    /// Spotlight isn't made active.
    private func bringToFront(_ window: NSWindow) {
        NSApp.setActivationPolicy(.regular)
        if let frontmost = NSWorkspace.shared.frontmostApplication,
            frontmost != NSRunningApplication.current
        {
            Self.logger.notice(
                "Activating from \(frontmost.bundleIdentifier ?? "?", privacy: .public)")
            NSRunningApplication.current.activate(from: frontmost, options: [.activateAllWindows])
        }
        NSApp.activate()
        window.makeKeyAndOrderFront(nil)
    }

    private func observe() {
        controller.$status
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in self?.model.exoStatus = status }
            .store(in: &cancellables)
        // Re-evaluated every second as well, so the grace period can end.
        localNetworkChecker.$status
            .combineLatest(Timer.publish(every: 1, on: .main, in: .common).autoconnect())
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.updateLocalNetwork() }
            .store(in: &cancellables)
    }

    private func updateLocalNetwork() {
        guard let askedAt else { return }
        let status = localNetworkChecker.status
        if status == .working {
            model.send(.localNetworkAllowed)
        } else if status.suggestsAccessOff {
            if Date().timeIntervalSince(askedAt) >= Self.localNetworkGracePeriod {
                model.send(.localNetworkDenied)
            }
        } else if case .notWorking = status {
            // Timed out or failed some other way: don't hold the user up.
            model.send(.localNetworkUndetermined)
        }
    }

    /// Continue on the first pane: the only place a new install is
    /// registered as a login item, and only if the box is ticked.
    private func recordLaunchAtLoginChoice() {
        preferences.chooseLaunchAtLogin(model.openAtLogin)
        LaunchAtLoginHelper.registerIfChosen(preferences: preferences)
        Self.logger.notice(
            "Open EXO at login: \(self.model.openAtLogin ? "on" : "off", privacy: .public)")
    }

    private func beginExo() {
        guard !exoStarted else { return }
        exoStarted = true
        askedAt = Date()
        startExo()
    }

    func windowWillClose(_ notification: Notification) {
        preferences.markWelcomeSeen()
        if model.flow.pane == .welcome {
            // Closed before choosing: nothing is registered without consent.
            preferences.markLaunchAtLoginDefaultApplied()
        }
        // Closing early still leaves EXO running, as on any later launch.
        beginExo()
        cancellables.removeAll()
        window = nil
        // Back to living in the menu bar only.
        NSApp.setActivationPolicy(.accessory)
    }
}
