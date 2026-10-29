import AppKit
import Combine
import os.log

/// EXO's menu bar extra: a template icon that shows EXO's state, and a real
/// menu (`NSMenu`) built from `MenuBarMenu` each time it opens.
///
/// AppKit rather than SwiftUI's `MenuBarExtra(.menu)`: status lines can be
/// secondary text, items can carry tooltips (the API URL, a model's repo
/// id), and the icon is only redrawn when its state changes, including the
/// starting pulse, which SwiftUI's menu bar label doesn't animate.
@MainActor
final class MenuBarController: NSObject, ObservableObject, NSMenuDelegate {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "MenuBar")
    static let localNetworkSettingsURL = URL(
        string: "x-apple.systempreferences:com.apple.preference.security?Privacy_LocalNetwork")!

    private let controller: ExoProcessController
    private let stateService: ClusterStateService
    private let localNetworkChecker: LocalNetworkChecker
    private let thunderboltBridgeService: ThunderboltBridgeService
    private let updater: SparkleUpdater
    private let bugReportWindowController: BugReportWindowController
    private let startupDashboardOpener: StartupDashboardOpener

    private var statusItem: NSStatusItem?
    private let menu = NSMenu()
    private var cancellables = Set<AnyCancellable>()
    private var iconStyle: MenuBarIconStyle?
    private var pulseTimer: Timer?
    private var pulseStart = Date()
    /// The snapshot the open menu was built from, for the quit warning.
    private var lastSnapshot = MenuBarSnapshot(exo: .stopped)

    init(
        controller: ExoProcessController, stateService: ClusterStateService,
        localNetworkChecker: LocalNetworkChecker,
        thunderboltBridgeService: ThunderboltBridgeService, updater: SparkleUpdater,
        bugReportWindowController: BugReportWindowController,
        startupDashboardOpener: StartupDashboardOpener
    ) {
        self.controller = controller
        self.stateService = stateService
        self.localNetworkChecker = localNetworkChecker
        self.thunderboltBridgeService = thunderboltBridgeService
        self.updater = updater
        self.bugReportWindowController = bugReportWindowController
        self.startupDashboardOpener = startupDashboardOpener
        super.init()
    }

    /// Puts the item in the menu bar.
    func install() {
        guard statusItem == nil else { return }
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        menu.delegate = self
        menu.autoenablesItems = false
        item.menu = menu
        statusItem = item

        // Redraw the icon only when what it shows changes. @Published
        // publishes before the property changes, so the style is worked out
        // on the next turn of the main queue, once it has.
        controller.$status.combineLatest(stateService.$apiState, localNetworkChecker.$accessIsOff)
            .combineLatest(
                thunderboltBridgeService.$detectedCycle, stateService.$localNodeId,
                updater.$availableUpdateVersion
            )
            .receive(on: DispatchQueue.main)
            .map { [weak self] _ in self?.currentIconStyle() }
            .compactMap { $0 }
            .removeDuplicates()
            .sink { [weak self] style in self?.show(style) }
            .store(in: &cancellables)

        // exo's API answering /node_id means it is up and the dashboard can
        // be opened.
        stateService.$localNodeId
            .compactMap { $0 }
            .sink { [weak self] _ in self?.startupDashboardOpener.exoIsUp() }
            .store(in: &cancellables)

        NSWorkspace.shared.notificationCenter
            .publisher(for: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification)
            .sink { [weak self] _ in
                guard let self, let style = self.iconStyle else { return }
                self.iconStyle = nil
                self.show(style)
            }
            .store(in: &cancellables)
    }

    // MARK: - Icon

    private func currentIconStyle() -> MenuBarIconStyle {
        let problemInMenu =
            localNetworkChecker.accessIsOff
            || thunderboltBridgeService.loopIncludes(nodeId: stateService.localNodeId)
        return MenuBarIconStyle.showing(
            controller.status, apiState: stateService.apiState, problemInMenu: problemInMenu,
            updateAvailable: updater.availableUpdateVersion != nil)
    }

    private var reduceMotion: Bool {
        NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
    }

    private func show(_ style: MenuBarIconStyle) {
        guard style != iconStyle else { return }
        iconStyle = style
        statusItem?.button?.setAccessibilityLabel(style.accessibilityLabel)
        statusItem?.button?.toolTip = style.accessibilityLabel
        pulseTimer?.invalidate()
        pulseTimer = nil
        drawIcon(phase: 0)
        guard style.opacity == .pulsing, !reduceMotion else { return }
        pulseStart = Date()
        let timer = Timer(timeInterval: 1.0 / 15, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                let period = 1.6
                let phase =
                    Date().timeIntervalSince(self.pulseStart)
                    .truncatingRemainder(dividingBy: period) / period
                self.drawIcon(phase: phase)
            }
        }
        // Keep pulsing while the menu is open.
        RunLoop.main.add(timer, forMode: .common)
        pulseTimer = timer
    }

    private func drawIcon(phase: Double) {
        guard let style = iconStyle else { return }
        statusItem?.button?.image = MenuBarGlyph.image(
            style: style, alpha: style.alpha(atPulsePhase: phase, reduceMotion: reduceMotion))
    }

    // MARK: - Menu

    func menuNeedsUpdate(_ menu: NSMenu) {
        lastSnapshot = snapshot()
        menu.removeAllItems()
        for entry in MenuBarMenu.entries(for: lastSnapshot) {
            menu.addItem(menuItem(for: entry))
        }
    }

    private func snapshot() -> MenuBarSnapshot {
        let localNodeId = stateService.localNodeId
        var snapshot = MenuBarSnapshot(exo: exoState())
        if controller.status == .running, let cluster = stateService.latestSnapshot {
            snapshot.devices = cluster.devices(localNodeId: localNodeId)
            snapshot.models = cluster.runningModels(
                localNodeId: localNodeId,
                displayNames: stateService.modelOptions.displayNames)
        }
        snapshot.localNetworkAccessOff = localNetworkChecker.accessIsOff
        let loopIncludesThisMac = thunderboltBridgeService.loopIncludes(nodeId: localNodeId)
        snapshot.thunderboltBridgeLoop = loopIncludesThisMac
        if (loopIncludesThisMac || ThunderboltLinks.anyLinkActive())
            && !NetworkSetupHelper.setupWasDeclined()
        {
            switch NetworkSetupHelper.setupState {
            case .current: snapshot.thunderboltNetworking = .fine
            case .notInstalled: snapshot.thunderboltNetworking = .notSetUp
            case .outdated: snapshot.thunderboltNetworking = .needsUpdate
            }
        }
        snapshot.updateVersion = updater.availableUpdateVersion
        return snapshot
    }

    private func exoState() -> MenuBarSnapshot.Exo {
        switch controller.status {
        case .stopped:
            return .stopped
        case .starting:
            return .starting
        case .running:
            return stateService.apiState == .notResponding ? .notResponding : .running
        case .failed(let message):
            return .failed(reason: message, wasRunning: controller.failedAfterRunning)
        }
    }

    private func menuItem(for entry: MenuEntry) -> NSMenuItem {
        switch entry {
        case .separator:
            return .separator()
        case .status(let text):
            let item = NSMenuItem(title: text, action: nil, keyEquivalent: "")
            item.isEnabled = false
            return item
        case .command(let title, let command, let enabled, let keyEquivalent, let toolTip):
            let item = NSMenuItem(
                title: title, action: #selector(menuItemChosen(_:)), keyEquivalent: keyEquivalent)
            item.target = self
            item.representedObject = CommandBox(command)
            item.isEnabled = enabled
            item.toolTip = toolTip
            return item
        case .submenu(let title, let entries, let enabled):
            let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
            let submenu = NSMenu(title: title)
            submenu.autoenablesItems = false
            for entry in entries {
                submenu.addItem(menuItem(for: entry))
            }
            item.submenu = submenu
            item.isEnabled = enabled
            return item
        }
    }

    @objc private func menuItemChosen(_ sender: NSMenuItem) {
        guard let box = sender.representedObject as? CommandBox else { return }
        run(box.command)
    }

    func run(_ command: MenuCommand) {
        switch command {
        case .installUpdate, .checkForUpdates:
            updater.checkForUpdates()
        case .openLocalNetworkSettings:
            NSWorkspace.shared.open(Self.localNetworkSettingsURL)
        case .turnOffThunderboltBridge:
            Task { await thunderboltBridgeService.disableThunderboltBridge() }
        case .setUpThunderboltNetworking:
            Task { await NetworkSetupHelper.offerSetUp() }
        case .openDashboard:
            openDashboard()
        case .copyAPIURL:
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(MenuBarMenu.apiURL, forType: .string)
        case .startExo:
            startExo()
        case .stopExo:
            stopExo()
        case .restartExo:
            stopExo()
            startExo()
        case .showLogs:
            controller.showLogs()
        case .openSettings:
            SettingsOpener.open()
        case .reportBug:
            bugReportWindowController.open()
        case .about:
            showAboutPanel()
        case .quit:
            quit()
        }
    }

    func openDashboard() {
        guard let url = URL(string: MenuBarMenu.dashboardURL) else { return }
        NSWorkspace.shared.open(url)
    }

    private func startExo() {
        stateService.resetTransientState()
        stateService.startPolling()
        controller.launchIfNeeded()
    }

    private func stopExo() {
        stateService.stopPolling()
        controller.stop()
        stateService.resetTransientState()
    }

    /// The standard About panel, with the build's commit as credits.
    private func showAboutPanel() {
        var options: [NSApplication.AboutPanelOptionKey: Any] = [:]
        let info = Bundle.main.infoDictionary ?? [:]
        if let tag = info["EXOBuildTag"] as? String, !tag.isEmpty {
            options[.applicationVersion] = tag
            options[.version] = ""
        }
        if let commit = info["EXOBuildCommit"] as? String, !commit.isEmpty {
            options[.credits] = NSAttributedString(
                string: "Commit \(commit.prefix(12))",
                attributes: [
                    .font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize),
                    .foregroundColor: NSColor.secondaryLabelColor,
                ])
        }
        NSApp.activate()
        NSApp.orderFrontStandardAboutPanel(options: options)
    }

    /// Quits, after asking if this Mac runs part of a model, which would
    /// stop.
    private func quit() {
        if let warning = MenuBarMenu.quitWarning(for: snapshot()) {
            let alert = NSAlert()
            alert.messageText = "Quit EXO?"
            alert.informativeText = warning
            alert.addButton(withTitle: "Quit")
            alert.addButton(withTitle: "Cancel")
            guard alert.runModalInFront() == .alertFirstButtonReturn else { return }
        }
        NSApp.terminate(nil)
    }

    /// EXO.app opened again while it runs (from Finder or Spotlight), maybe
    /// because its icon is hidden: show something.
    func handleReopen() {
        if controller.status == .running {
            openDashboard()
        } else {
            SettingsOpener.open()
        }
    }
}

/// Carries a `MenuCommand` in `NSMenuItem.representedObject`.
private final class CommandBox: NSObject {
    let command: MenuCommand

    init(_ command: MenuCommand) {
        self.command = command
    }
}
