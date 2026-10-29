import AppKit
import Foundation
import Testing

@testable import EXO

// MARK: - Icon

struct MenuBarIconStyleTests {

    @Test func runningIsSolidAndStoppedIsDimmed() {
        let running = MenuBarIconStyle.showing(
            .running, apiState: .responding, problemInMenu: false, updateAvailable: false)
        #expect(running.opacity == .full)
        #expect(!running.needsAttention && !running.updateAvailable)
        #expect(running.accessibilityLabel == "EXO, running")

        let stopped = MenuBarIconStyle.showing(
            .stopped, apiState: .notResponding, problemInMenu: true, updateAvailable: false)
        #expect(stopped.opacity == .dimmed)
        #expect(!stopped.needsAttention)  // nothing to fix while EXO is off
        #expect(stopped.alpha(atPulsePhase: 0, reduceMotion: false) == 0.35)
    }

    @Test func startingPulsesUnlessReduceMotionIsOn() {
        let starting = MenuBarIconStyle.showing(
            .starting, apiState: .connecting, problemInMenu: false, updateAvailable: false)
        #expect(starting.opacity == .pulsing)
        #expect(starting.accessibilityLabel == "EXO, starting")
        let low = starting.alpha(atPulsePhase: 0, reduceMotion: false)
        let high = starting.alpha(atPulsePhase: 0.5, reduceMotion: false)
        #expect(abs(low - 0.35) < 0.001 && abs(high - 1) < 0.001)
        for phase in [0, 0.25, 0.5, 0.9] {
            #expect(starting.alpha(atPulsePhase: phase, reduceMotion: true) == 0.6)
        }
    }

    @Test func failuresAndProblemsGetABadgeNotJustAColour() {
        let failed = MenuBarIconStyle.showing(
            .failed(message: "exo quit with an error (exit code 1)."), apiState: .responding,
            problemInMenu: false, updateAvailable: false)
        #expect(failed.opacity == .dimmed && failed.needsAttention)
        #expect(failed.accessibilityLabel == "EXO, couldn't start")

        let unresponsive = MenuBarIconStyle.showing(
            .running, apiState: .notResponding, problemInMenu: false, updateAvailable: false)
        #expect(unresponsive.opacity == .full && unresponsive.needsAttention)
        #expect(unresponsive.accessibilityLabel == "EXO, not responding")

        let localNetworkOff = MenuBarIconStyle.showing(
            .running, apiState: .responding, problemInMenu: true, updateAvailable: true)
        #expect(localNetworkOff.needsAttention && localNetworkOff.updateAvailable)
        #expect(
            localNetworkOff.accessibilityLabel
                == "EXO, running, needs attention, update available")
    }
}

struct MenuBarGlyphTests {

    /// Renders the icon at 2x and returns the alpha (0...255) at a point,
    /// measured from the top left in points.
    private func alpha(of image: NSImage, x: CGFloat, y: CGFloat) -> UInt8 {
        let scale: CGFloat = 2
        let width = Int(image.size.width * scale)
        let height = Int(image.size.height * scale)
        let bitmap = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8,
            samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
            bytesPerRow: 0, bitsPerPixel: 0)!
        bitmap.size = image.size
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
        image.draw(in: NSRect(origin: .zero, size: image.size))
        NSGraphicsContext.restoreGraphicsState()
        let color = bitmap.colorAt(x: Int(x * scale), y: Int(y * scale))!
        return UInt8((color.alphaComponent * 255).rounded())
    }

    private func style(
        _ status: ExoProcessController.Status, problem: Bool = false, update: Bool = false
    ) -> MenuBarIconStyle {
        .showing(status, apiState: .responding, problemInMenu: problem, updateAvailable: update)
    }

    @Test func wordmarkIsTheLogosTwentyByFiveGrid() {
        #expect(MenuBarGlyph.wordmark.count == 5)
        #expect(MenuBarGlyph.wordmark.allSatisfy { $0.count == 20 })
        #expect(MenuBarGlyph.wordmarkRect.size == NSSize(width: 40, height: 10))
    }

    @Test func iconIsATemplateImageThatFitsTheMenuBar() {
        let image = MenuBarGlyph.image(style: style(.running), alpha: 1)
        #expect(image.isTemplate)
        #expect(image.size.height <= 22)
    }

    @Test func runningIsOpaqueAndStoppedIsDimmed() {
        // The first pixel of the "e"'s left stem, row 2 from the top.
        let rect = MenuBarGlyph.wordmarkRect
        let stem = (x: rect.minX + 1, y: (MenuBarGlyph.size.height - rect.maxY) + 5)
        let running = MenuBarGlyph.image(style: style(.running), alpha: 1)
        #expect(alpha(of: running, x: stem.x, y: stem.y) == 255)
        let stopped = MenuBarGlyph.image(style: style(.stopped), alpha: 0.35)
        #expect(abs(Int(alpha(of: stopped, x: stem.x, y: stem.y)) - 89) <= 2)
        // A gap in the grid stays clear: the top row's first pixel is empty.
        #expect(alpha(of: running, x: rect.minX + 1, y: stem.y - 4) == 0)
    }

    @Test func badgesAreShapesCutOutOfTheWordmark() {
        let plain = MenuBarGlyph.image(style: style(.running), alpha: 1)
        let badged = MenuBarGlyph.image(style: style(.running, problem: true), alpha: 1)
        let dotted = MenuBarGlyph.image(style: style(.running, update: true), alpha: 1)
        let badge = MenuBarGlyph.attentionBadgeRect
        // Inside the badge's circle, below the wordmark and left of the "!".
        let inBadge = (x: badge.midX - 2, y: MenuBarGlyph.size.height - (badge.minY + 2))
        #expect(alpha(of: plain, x: inBadge.x, y: inBadge.y) == 0)
        #expect(alpha(of: badged, x: inBadge.x, y: inBadge.y) > 200)
        // The exclamation mark is cut out of the badge.
        let bar = MenuBarGlyph.exclamationBar
        #expect(alpha(of: badged, x: bar.midX, y: MenuBarGlyph.size.height - bar.midY) == 0)
        let dot = MenuBarGlyph.updateDotRect
        let dotCentre = (x: dot.midX, y: MenuBarGlyph.size.height - dot.midY)
        #expect(alpha(of: plain, x: dotCentre.x, y: dotCentre.y) == 0)
        #expect(alpha(of: dotted, x: dotCentre.x, y: dotCentre.y) == 255)
        // A failed EXO's badge stays opaque on the dimmed wordmark.
        let failed = MenuBarGlyph.image(
            style: style(.failed(message: "x")), alpha: MenuBarIconStyle.dimmedAlpha)
        #expect(alpha(of: failed, x: inBadge.x, y: inBadge.y) > 200)
    }
}

// MARK: - Menu

private let thisMac = ClusterDevice(
    id: "a", name: "Mac Studio", isThisMac: true, usedMemoryBytes: 21 << 30,
    totalMemoryBytes: 64 << 30)
private let otherMac = ClusterDevice(
    id: "b", name: "Mac mini", isThisMac: false, usedMemoryBytes: 12 << 30,
    totalMemoryBytes: 64 << 30)
private let llama = RunningModel(
    id: "i1", displayName: "Llama 3.2 3B",
    repoId: "mlx-community/Llama-3.2-3B-Instruct-4bit", status: .ready, runsOnThisMac: true)

private func titles(_ entries: [MenuEntry]) -> [String] {
    entries.map { entry in
        switch entry {
        case .status(let text): return text
        case .command(let title, _, _, _, _): return title
        case .submenu(let title, _, _): return title + " ▸"
        case .separator: return "—"
        }
    }
}

private func command(_ title: String, in entries: [MenuEntry]) -> MenuEntry? {
    entries.first {
        if case .command(title, _, _, _, _) = $0 { return true }
        return false
    }
}

private func isEnabled(_ entry: MenuEntry?) -> Bool {
    switch entry {
    case .command(_, _, let enabled, _, _): return enabled
    case .submenu(_, _, let enabled): return enabled
    default: return false
    }
}

struct MenuBarMenuTests {

    @Test func runningInAClusterReadsLikeTheSpec() {
        let snapshot = MenuBarSnapshot(
            exo: .running, devices: [thisMac, otherMac], models: [llama])
        #expect(
            titles(MenuBarMenu.entries(for: snapshot)) == [
                "EXO is running · 2 devices", "21 of 64 GB in use on this Mac",
                "Models ▸", "Devices ▸",
                "—", "Open Dashboard", "Copy API URL",
                "—", "Stop EXO", "Show Logs",
                "—", "Settings…", "Check for Updates…", "Report a Bug…",
                "—", "About EXO", "Quit EXO",
            ])
    }

    @Test func submenusShowModelsByNameAndDevicesWithMemory() {
        let snapshot = MenuBarSnapshot(
            exo: .running, devices: [thisMac, otherMac], models: [llama])
        let entries = MenuBarMenu.entries(for: snapshot)
        guard case .submenu(_, let models, true) = entries[2],
            case .submenu(_, let devices, true) = entries[3]
        else {
            Issue.record("expected enabled Models and Devices submenus")
            return
        }
        #expect(
            titles(models) == ["Llama 3.2 3B — Ready", "—", "Manage Models in Dashboard"])
        if case .command(_, .openDashboard, _, _, let toolTip) = models[0] {
            #expect(toolTip == "mlx-community/Llama-3.2-3B-Instruct-4bit")
        }
        #expect(
            titles(devices) == ["Mac Studio (this Mac) — 21 of 64 GB", "Mac mini — 12 of 64 GB"])
    }

    @Test func dashboardAndAPIURLAreDisabledNotHiddenUnlessRunning() {
        for exo in [
            MenuBarSnapshot.Exo.stopped, .starting, .notResponding,
            .failed(reason: "x", wasRunning: false),
        ] {
            let entries = MenuBarMenu.entries(for: MenuBarSnapshot(exo: exo))
            #expect(!isEnabled(command("Open Dashboard", in: entries)))
            #expect(!isEnabled(command("Copy API URL", in: entries)))
            #expect(command("Show Logs", in: entries) != nil)
        }
        let running = MenuBarMenu.entries(for: MenuBarSnapshot(exo: .running))
        #expect(isEnabled(command("Open Dashboard", in: running)))
        if case .command(_, .copyAPIURL, _, _, let toolTip) = command("Copy API URL", in: running) {
            #expect(toolTip == "http://localhost:52415/v1")
        }
    }

    @Test(
        arguments: [
            (MenuBarSnapshot.Exo.stopped, "EXO is stopped", "Start EXO"),
            (.starting, "EXO is starting…", "Stop EXO"),
            (.running, "EXO is running on this Mac", "Stop EXO"),
            (.notResponding, "EXO isn't responding", "Restart EXO"),
            (
                .failed(reason: "exo crashed (signal 11).", wasRunning: true),
                "EXO stopped unexpectedly", "Restart EXO"
            ),
            (
                .failed(reason: "exo crashed (signal 11).", wasRunning: false),
                "EXO couldn't start", "Restart EXO"
            ),
        ])
    func statusLineAndRunControlFollowTheState(
        exo: MenuBarSnapshot.Exo, status: String, control: String
    ) {
        let entries = MenuBarMenu.entries(for: MenuBarSnapshot(exo: exo))
        #expect(titles(entries).first == status)
        #expect(titles(entries).contains(control))
    }

    @Test func aFailureSaysWhyOnWrappedLines() {
        let reason =
            "Another exo is already running on this Mac (PID 4321). Quit it, then choose Restart EXO."
        let lines = MenuBarMenu.statusLines(
            for: MenuBarSnapshot(exo: .failed(reason: reason, wasRunning: false)))
        #expect(lines.first == "EXO couldn't start")
        #expect(lines.dropFirst().joined(separator: " ") == reason)
        #expect(lines.allSatisfy { $0.count <= MenuBarMenu.statusLineLength })
    }

    @Test func fixItRowsAppearOnlyWhenNeeded() {
        #expect(MenuBarMenu.fixItRows(for: MenuBarSnapshot(exo: .running)).isEmpty)

        var snapshot = MenuBarSnapshot(exo: .running)
        snapshot.localNetworkAccessOff = true
        snapshot.thunderboltBridgeLoop = true
        snapshot.thunderboltNetworking = .notSetUp
        snapshot.needsRestart = true
        snapshot.updateVersion = "1.0.73"
        #expect(
            titles(MenuBarMenu.fixItRows(for: snapshot)) == [
                "Local Network Access Is Off", "Open Privacy & Security Settings…",
                "Thunderbolt Bridge Loop Detected", "Turn Off Thunderbolt Bridge…",
                "Thunderbolt Networking Isn't Set Up", "Set Up Thunderbolt Networking…",
                "Restart EXO to Apply Settings", "Install Update (EXO 1.0.73)…",
            ])

        snapshot.thunderboltNetworking = .needsUpdate
        #expect(
            titles(MenuBarMenu.fixItRows(for: snapshot)).contains("Update Thunderbolt Networking…"))

        // Local Network access only matters while EXO is on.
        snapshot.exo = .stopped
        #expect(
            !titles(MenuBarMenu.fixItRows(for: snapshot)).contains("Local Network Access Is Off"))
    }

    @Test func changedSettingsOfferARestartUntilExoHasThem() {
        var snapshot = MenuBarSnapshot(exo: .running)
        #expect(
            command("Restart EXO to Apply Settings", in: MenuBarMenu.entries(for: snapshot)) == nil)

        snapshot.needsRestart = true
        let entries = MenuBarMenu.entries(for: snapshot)
        guard
            case .command(_, let action, let enabled, _, _) = command(
                "Restart EXO to Apply Settings", in: entries)
        else {
            Issue.record("no Restart EXO to Apply Settings item")
            return
        }
        #expect(action == .restartToApplySettings)
        #expect(enabled)
        // A fix-it row: after the status lines and submenus, before Open Dashboard.
        let restart = titles(entries).firstIndex(of: "Restart EXO to Apply Settings") ?? 0
        let dashboard = titles(entries).firstIndex(of: "Open Dashboard") ?? 0
        #expect(restart < dashboard)
    }

    @Test func settingsAndQuitHaveTheirStandardShortcuts() {
        let entries = MenuBarMenu.entries(for: MenuBarSnapshot(exo: .running))
        if case .command(_, .openSettings, _, let key, _) = command("Settings…", in: entries) {
            #expect(key == ",")
        } else {
            Issue.record("no Settings… item")
        }
        if case .command(_, .quit, _, let key, _) = command("Quit EXO", in: entries) {
            #expect(key == "q")
        } else {
            Issue.record("no Quit EXO item")
        }
    }

    @Test func quittingWarnsOnlyWhenThisMacRunsAModel() {
        #expect(MenuBarMenu.quitWarning(for: MenuBarSnapshot(exo: .running)) == nil)
        let elsewhere = RunningModel(
            id: "i2", displayName: "Qwen3 30B", repoId: "q", status: .ready, runsOnThisMac: false)
        #expect(
            MenuBarMenu.quitWarning(for: MenuBarSnapshot(exo: .running, models: [elsewhere]))
                == nil)
        #expect(
            MenuBarMenu.quitWarning(
                for: MenuBarSnapshot(exo: .running, models: [llama, elsewhere]))
                == "Llama 3.2 3B runs on this Mac and will stop.")
        #expect(
            MenuBarMenu.quitWarning(for: MenuBarSnapshot(exo: .stopped, models: [llama])) == nil)
    }
}

// MARK: - Cluster state to menu rows

private func decodeState(_ json: String) throws -> ClusterState {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(ClusterState.self, from: Data(json.utf8))
}

private let stateJSON = """
    {
      "instances": {
        "i1": {"MlxRingInstance": {"instanceId": "i1", "shardAssignments": {
          "modelId": "mlx-community/Llama-3.2-3B-Instruct-4bit",
          "nodeToRunner": {"this": "r1", "other": "r2"}}}},
        "i2": {"MlxRingInstance": {"instanceId": "i2", "shardAssignments": {
          "modelId": "mlx-community/Qwen3-30B-A3B-4bit",
          "nodeToRunner": {"other": "r3"}}}}
      },
      "runners": {
        "r1": {"RunnerReady": {}}, "r2": {"RunnerRunning": {}}, "r3": {"RunnerIdle": {}}
      },
      "downloads": {
        "other": [
          {"DownloadOngoing": {"nodeId": "other",
            "shardMetadata": {"PipelineShardMetadata": {"modelCard": {"modelId": "mlx-community/Qwen3-30B-A3B-4bit"}}},
            "downloadProgress": {"totalBytes": {"inBytes": 1000}, "downloadedBytes": {"inBytes": 425}}}},
          {"DownloadOngoing": {"nodeId": "other",
            "shardMetadata": {"PipelineShardMetadata": {"modelCard": {"modelId": "mlx-community/Some-Other-Model"}}},
            "downloadProgress": {"totalBytes": {"inBytes": 1000}, "downloadedBytes": {"inBytes": 1}}}}
        ]
      },
      "nodeIdentities": {
        "this": {"modelId": "Mac Studio", "chipId": "M3 Ultra", "friendlyName": "Studio"},
        "other": {"modelId": "Mac mini", "chipId": "M4 Pro", "friendlyName": "Unknown"}
      },
      "nodeMemory": {
        "this": {"ramTotal": {"inBytes": 206158430208}, "ramAvailable": {"inBytes": 162135015424}},
        "other": {"ramTotal": {"inBytes": 68719476736}, "ramAvailable": {"inBytes": 55834574848}}
      }
    }
    """

struct ClusterSummaryTests {

    @Test func runningModelsUseDisplayNamesAndTheGlossaryStatuses() throws {
        let state = try decodeState(stateJSON)
        let models = state.runningModels(
            localNodeId: "this",
            displayNames: ["mlx-community/Llama-3.2-3B-Instruct-4bit": "Llama 3.2 3B"])
        #expect(models.map(\.displayName) == ["Llama 3.2 3B", "Qwen3-30B-A3B-4bit"])
        #expect(models.map(\.status) == [.ready, .downloading(percent: 42)])
        #expect(models.map(\.runsOnThisMac) == [true, false])
        #expect(models[1].status.label == "Downloading 42%")
    }

    @Test(
        arguments: [
            ([], nil, RunningModel.Status.starting),
            (["RunnerReady", "RunnerLoading"], nil, .starting),
            (["RunnerReady", "RunnerRunning"], nil, .ready),
            (["RunnerLoaded"], nil, .ready),
            (["RunnerReady", "RunnerFailed"], nil, .failed),
            (["RunnerShuttingDown", "RunnerReady"], nil, .stopping),
            (["RunnerIdle"], 0.999, .downloading(percent: 99)),
            (["RunnerFailed"], 0.5, .failed),
        ] as [([String], Double?, RunningModel.Status)])
    func modelStatusMatchesTheDashboard(
        runners: [String], downloaded: Double?, expected: RunningModel.Status
    ) {
        #expect(
            RunningModel.Status(runnerStatuses: runners, downloadFraction: downloaded) == expected)
    }

    @Test func devicesListThisMacFirstWithMemoryInUse() throws {
        let devices = try decodeState(stateJSON).devices(localNodeId: "this")
        #expect(devices.map(\.name) == ["Studio", "Mac mini"])
        #expect(devices.map(\.isThisMac) == [true, false])
        #expect(devices.map(\.memoryInUse) == ["41 of 192 GB", "12 of 64 GB"])
    }

    @Test func stateWithChatTasksStillDecodes() throws {
        // The menu bar no longer shows prompts, so the state's tasks (which
        // carry other people's messages) aren't read at all.
        let withTasks = stateJSON.replacingOccurrences(
            of: "\"runners\": {",
            with: "\"tasks\": {\"t\": {\"TextGeneration\": {\"taskId\": \"t\"}}}, \"runners\": {")
        #expect(try decodeState(withTasks).instances.count == 2)
    }
}

struct ModelNamesTests {

    @Test func theModelListIsFetchedAgainWhileARunningModelIsMissing() {
        let now = Date()
        let llamaOption = ModelOption(id: "llama", displayName: "Llama 3.2 1B")
        #expect(
            ClusterStateService.shouldFetchModels(running: [], known: [], lastFetch: nil, now: now))
        #expect(
            !ClusterStateService.shouldFetchModels(
                running: ["llama"], known: [llamaOption], lastFetch: nil, now: now))
        // A model exo hadn't loaded a card for when the list was fetched.
        #expect(
            ClusterStateService.shouldFetchModels(
                running: ["llama", "qwen"], known: [llamaOption],
                lastFetch: now.addingTimeInterval(-6), now: now))
        // But not more than every 5 seconds.
        #expect(
            !ClusterStateService.shouldFetchModels(
                running: ["qwen"], known: [llamaOption], lastFetch: now.addingTimeInterval(-1),
                now: now))
    }
}

struct LocalNetworkAccessTests {

    @Test func accessCountsAsOffOnlyAfterTwoChecksInARow() {
        typealias Status = LocalNetworkChecker.Status
        let denied = Status.notWorking(reason: Status.deniedByMacOSReason)
        let unreachable = Status.notWorking(reason: Status.deniedReason)
        let timedOut = Status.notWorking(reason: "Timeout (may be blocked)")
        // macOS reports "off" for a moment right after exo starts.
        #expect(!LocalNetworkChecker.accessIsOff(current: denied, previous: .checking))
        #expect(LocalNetworkChecker.accessIsOff(current: denied, previous: denied))
        #expect(!LocalNetworkChecker.accessIsOff(current: unreachable, previous: .working))
        #expect(LocalNetworkChecker.accessIsOff(current: unreachable, previous: denied))
        #expect(!LocalNetworkChecker.accessIsOff(current: timedOut, previous: timedOut))
        #expect(!LocalNetworkChecker.accessIsOff(current: .working, previous: denied))
    }
}
