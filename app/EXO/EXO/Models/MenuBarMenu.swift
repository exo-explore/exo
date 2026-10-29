import Foundation

/// What the menu bar menu needs to know, gathered when it opens.
struct MenuBarSnapshot: Equatable {
    enum Exo: Equatable {
        case stopped
        case starting
        case running
        case notResponding
        /// exo exited or couldn't be started. `wasRunning` tells a crash
        /// from a failed start.
        case failed(reason: String, wasRunning: Bool)
    }

    /// Thunderbolt networking (EXO's network service), when it matters.
    enum ThunderboltNetworking: Equatable {
        case fine
        case notSetUp
        case needsUpdate
    }

    var exo: Exo
    /// This Mac first.
    var devices: [ClusterDevice] = []
    var models: [RunningModel] = []
    var localNetworkAccessOff = false
    /// A Thunderbolt Bridge loop that this Mac is part of.
    var thunderboltBridgeLoop = false
    var thunderboltNetworking: ThunderboltNetworking = .fine
    var updateVersion: String?
}

/// Something a menu item does.
enum MenuCommand: Equatable {
    case installUpdate
    case openLocalNetworkSettings
    case turnOffThunderboltBridge
    case setUpThunderboltNetworking
    case openDashboard
    case copyAPIURL
    case startExo
    case stopExo
    case restartExo
    case showLogs
    case openSettings
    case checkForUpdates
    case reportBug
    case about
    case quit
}

/// One row of the menu.
indirect enum MenuEntry: Equatable {
    /// A line of status: shown as a disabled item.
    case status(String)
    case command(
        String, MenuCommand, enabled: Bool = true, keyEquivalent: String = "",
        toolTip: String? = nil)
    case submenu(String, [MenuEntry], enabled: Bool)
    case separator
}

/// The menu bar menu: a real menu, laid out as in the design spec.
enum MenuBarMenu {
    static let apiURL = "http://localhost:52415/v1"
    static let dashboardURL = "http://localhost:52415/"
    /// Status lines longer than this wrap onto another line.
    static let statusLineLength = 46

    static func entries(for snapshot: MenuBarSnapshot) -> [MenuEntry] {
        var entries = statusLines(for: snapshot).map(MenuEntry.status)
        let isRunning = snapshot.exo == .running
        entries.append(.submenu("Models", modelsSubmenu(snapshot), enabled: isRunning))
        entries.append(.submenu("Devices", devicesSubmenu(snapshot), enabled: isRunning))

        let fixes = fixItRows(for: snapshot)
        if !fixes.isEmpty {
            entries.append(.separator)
            entries += fixes
        }

        entries += [
            .separator,
            .command("Open Dashboard", .openDashboard, enabled: isRunning),
            .command("Copy API URL", .copyAPIURL, enabled: isRunning, toolTip: apiURL),
            .separator,
            runControl(for: snapshot.exo),
            .command("Show Logs", .showLogs),
            .separator,
            .command("Settings…", .openSettings, keyEquivalent: ","),
            .command("Check for Updates…", .checkForUpdates),
            .command("Report a Bug…", .reportBug),
            .separator,
            .command("About EXO", .about),
            .command("Quit EXO", .quit, keyEquivalent: "q"),
        ]
        return entries
    }

    static func statusLines(for snapshot: MenuBarSnapshot) -> [String] {
        switch snapshot.exo {
        case .stopped:
            return ["EXO is stopped"]
        case .starting:
            return ["EXO is starting…"]
        case .notResponding:
            return ["EXO isn't responding"]
        case .failed(let reason, let wasRunning):
            let headline = wasRunning ? "EXO stopped unexpectedly" : "EXO couldn't start"
            return [headline] + wrapped(reason, width: statusLineLength)
        case .running:
            let otherDevices = snapshot.devices.filter { !$0.isThisMac }.count
            var lines = [
                otherDevices == 0
                    ? "EXO is running on this Mac"
                    : "EXO is running · \(otherDevices + 1) devices"
            ]
            if let memory = snapshot.devices.first(where: \.isThisMac)?.memoryInUse {
                lines.append("\(memory) in use on this Mac")
            }
            return lines
        }
    }

    private static func modelsSubmenu(_ snapshot: MenuBarSnapshot) -> [MenuEntry] {
        let models: [MenuEntry] =
            snapshot.models.isEmpty
            ? [.status("No Models Running")]
            : snapshot.models.map {
                .command(
                    "\($0.displayName) — \($0.status.label)", .openDashboard, toolTip: $0.repoId)
            }
        return models + [.separator, .command("Manage Models in Dashboard", .openDashboard)]
    }

    private static func devicesSubmenu(_ snapshot: MenuBarSnapshot) -> [MenuEntry] {
        guard !snapshot.devices.isEmpty else { return [.status("No Devices")] }
        return snapshot.devices.map { device in
            let name = device.isThisMac ? "\(device.name) (this Mac)" : device.name
            return .status(device.memoryInUse.map { "\(name) — \($0)" } ?? name)
        }
    }

    /// Problems the user can fix, each a status line and the action that
    /// fixes it.
    static func fixItRows(for snapshot: MenuBarSnapshot) -> [MenuEntry] {
        var rows: [MenuEntry] = []
        if snapshot.localNetworkAccessOff, snapshot.exo != .stopped {
            rows += [
                .status("Local Network Access Is Off"),
                .command("Open Privacy & Security Settings…", .openLocalNetworkSettings),
            ]
        }
        if snapshot.thunderboltBridgeLoop {
            rows += [
                .status("Thunderbolt Bridge Loop Detected"),
                .command("Turn Off Thunderbolt Bridge…", .turnOffThunderboltBridge),
            ]
        }
        switch snapshot.thunderboltNetworking {
        case .fine:
            break
        case .notSetUp:
            rows += [
                .status("Thunderbolt Networking Isn't Set Up"),
                .command("Set Up Thunderbolt Networking…", .setUpThunderboltNetworking),
            ]
        case .needsUpdate:
            rows += [
                .status("Thunderbolt Networking Needs an Update"),
                .command("Update Thunderbolt Networking…", .setUpThunderboltNetworking),
            ]
        }
        if let version = snapshot.updateVersion {
            rows.append(.command("Install Update (EXO \(version))…", .installUpdate))
        }
        return rows
    }

    /// One item whose label says what it will do.
    static func runControl(for exo: MenuBarSnapshot.Exo) -> MenuEntry {
        switch exo {
        case .stopped:
            return .command("Start EXO", .startExo)
        case .starting, .running:
            return .command("Stop EXO", .stopExo)
        case .notResponding, .failed:
            return .command("Restart EXO", .restartExo)
        }
    }

    /// What quitting stops, if this Mac runs part of a model: asked before
    /// quitting. Nil when nothing would stop.
    static func quitWarning(for snapshot: MenuBarSnapshot) -> String? {
        guard snapshot.exo == .running || snapshot.exo == .starting else { return nil }
        let names = snapshot.models.filter(\.runsOnThisMac).map(\.displayName)
        switch names.count {
        case 0:
            return nil
        case 1:
            return "\(names[0]) runs on this Mac and will stop."
        case 2:
            return "\(names[0]) and \(names[1]) run on this Mac and will stop."
        default:
            return "\(names[0]) and \(names.count - 1) other models run on this Mac and will stop."
        }
    }

    /// Splits `text` into lines of at most `width` characters, at spaces.
    static func wrapped(_ text: String, width: Int) -> [String] {
        var lines: [String] = []
        var line = ""
        for word in text.split(separator: " ") {
            if line.isEmpty {
                line = String(word)
            } else if line.count + 1 + word.count <= width {
                line += " " + word
            } else {
                lines.append(line)
                line = String(word)
            }
        }
        if !line.isEmpty { lines.append(line) }
        return lines
    }
}
