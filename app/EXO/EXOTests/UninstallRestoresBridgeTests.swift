import CryptoKit
import Foundation
import Testing

@testable import EXO

/// Runs the uninstall script's Thunderbolt Bridge step under /bin/sh (as
/// AppleScript's `do shell script` does), with stand-ins for networksetup,
/// scutil and PlistBuddy that print what a Mac would and record their calls.
/// The network preferences path is redirected, so nothing on the machine is
/// read or changed.
private final class BridgeSandbox {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("EXOTests-bridge-\(UUID().uuidString)", isDirectory: true)
    private var bin: URL { root.appendingPathComponent("bin", isDirectory: true) }
    private var fixtures: URL { root.appendingPathComponent("fixtures", isDirectory: true) }
    private var callLog: URL { root.appendingPathComponent("calls.log") }
    private var scutilInput: URL { root.appendingPathComponent("scutil.stdin") }

    /// What `networksetup -listnetworkserviceorder` prints: Ethernet, Wi-Fi
    /// and a disabled Thunderbolt Bridge whose device is bridge0.
    static let serviceOrder = """
        An asterisk (*) denotes that a network service is disabled.
        (1) Ethernet
        (Hardware Port: Ethernet, Device: en0)

        (2) Wi-Fi
        (Hardware Port: Wi-Fi, Device: en1)

        (*) Thunderbolt Bridge
        (Hardware Port: Thunderbolt Bridge, Device: bridge0)

        """

    static let hardwarePorts = """

        Hardware Port: Ethernet
        Device: en0
        Ethernet Address: 00:00:00:00:00:01

        Hardware Port: Wi-Fi
        Device: en1
        Ethernet Address: 00:00:00:00:00:02

        Hardware Port: Thunderbolt 1
        Device: en2
        Ethernet Address: 00:00:00:00:00:03

        Hardware Port: Thunderbolt 2
        Device: en3
        Ethernet Address: 00:00:00:00:00:04

        Hardware Port: Thunderbolt Bridge
        Device: bridge0
        Ethernet Address: 00:00:00:00:00:05

        VLAN Configurations
        ===================

        """

    init(serviceOrder: String = serviceOrder, bridgeDefined: Bool) throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: bin, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: fixtures, withIntermediateDirectories: true)
        try "".write(to: callLog, atomically: true, encoding: .utf8)
        try serviceOrder.write(
            to: fixtures.appendingPathComponent("serviceorder"), atomically: true, encoding: .utf8)
        try Self.hardwarePorts.write(
            to: fixtures.appendingPathComponent("hardwareports"), atomically: true, encoding: .utf8)
        try "Ethernet\nWi-Fi\n*Thunderbolt Bridge\n".write(
            to: fixtures.appendingPathComponent("services"), atomically: true, encoding: .utf8)

        let stubs = [
            "networksetup": """
            echo "networksetup $*" >> "\(callLog.path)"
            case "$1" in
              -listnetworkserviceorder) cat "\(fixtures.path)/serviceorder" ;;
              -listallhardwareports) cat "\(fixtures.path)/hardwareports" ;;
              -listallnetworkservices) cat "\(fixtures.path)/services" ;;
            esac
            """,
            "scutil": """
            echo "scutil $*" >> "\(callLog.path)"
            cat >> "\(scutilInput.path)"
            """,
            "ifconfig": "echo \"ifconfig $*\" >> \"\(callLog.path)\"",
            "PlistBuddy": "exit \(bridgeDefined ? 0 : 1)",
        ]
        for (command, body) in stubs {
            let stub = bin.appendingPathComponent(command)
            try "#!/bin/bash\n\(body)\n".write(to: stub, atomically: true, encoding: .utf8)
            try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: stub.path)
        }
    }

    deinit {
        try? FileManager.default.removeItem(at: root)
    }

    /// The restore step with every system path redirected into the sandbox.
    private var sandboxedScript: String {
        let script =
            "set -euo pipefail\n"
            + NetworkSetupHelper.restoreThunderboltBridgeScript
            .replacingOccurrences(
                of: "/Library/Preferences/SystemConfiguration/preferences.plist",
                with: root.appendingPathComponent("preferences.plist").path
            )
            .replacingOccurrences(
                of: "/usr/libexec/PlistBuddy", with: bin.appendingPathComponent("PlistBuddy").path)
        #expect(!script.contains("/Library/") && !script.contains("/usr/libexec/"))
        return script
    }

    /// Runs the step the way Uninstall EXO does, through AppleScript's
    /// `do shell script` and the app's escaping, but without administrator
    /// privileges. Returns AppleScript's error message, if any.
    @MainActor
    func runThroughAppleScript() -> String? {
        let script = "PATH=\"\(bin.path):/usr/bin:/bin\"; export PATH\n" + sandboxedScript
        let source = NetworkSetupHelper.appleScriptSource(running: script, asAdministrator: false)
        var errorInfo: NSDictionary?
        NSAppleScript(source: source)?.executeAndReturnError(&errorInfo)
        return errorInfo.map { ($0[NSAppleScript.errorMessage] as? String) ?? "failed" }
    }

    func run() throws -> Int32 {
        let script = sandboxedScript
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", script]
        process.environment = ["PATH": "\(bin.path):/usr/bin:/bin"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
    }

    var calls: [String] {
        ((try? String(contentsOf: callLog, encoding: .utf8)) ?? "")
            .split(separator: "\n").map(String.init)
    }

    var scutilCommands: [String] {
        ((try? String(contentsOf: scutilInput, encoding: .utf8)) ?? "")
            .split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
    }
}

struct UninstallRestoresBridgeTests {

    @Test func recreatesTheBridgeTheSetupDeletedAndEnablesIt() throws {
        let sandbox = try BridgeSandbox(bridgeDefined: false)

        #expect(try sandbox.run() == 0)

        #expect(
            sandbox.scutilCommands == [
                "d.init",
                "d.add Interfaces * en2 en3",
                "set /VirtualNetworkInterfaces/Bridge/bridge0",
                "d.init",
                "d.add __AUTO__ thunderbolt-bridge",
                "set /VirtualNetworkInterfaces/Bridge/bridge0/Options",
                "commit",
                "apply",
                "quit",
            ])
        // The ports are brought up first: configd can't add a port that is
        // down (seen on a Mac Studio after leaving the exo location).
        let calls = sandbox.calls
        let bringUp = calls.firstIndex(of: "ifconfig en3 up")
        let writeBridge = calls.firstIndex(of: "scutil --prefs")
        #expect(calls.contains("ifconfig en2 up"))
        #expect(bringUp != nil && writeBridge != nil && bringUp! < writeBridge!)
        #expect(calls.contains("networksetup -setnetworkserviceenabled Thunderbolt Bridge on"))
    }

    @MainActor
    @Test func survivesTheAppleScriptEscapingUninstallUses() throws {
        let sandbox = try BridgeSandbox(bridgeDefined: false)

        #expect(sandbox.runThroughAppleScript() == nil)

        #expect(sandbox.scutilCommands.contains("d.add Interfaces * en2 en3"))
        #expect(
            sandbox.calls.contains("networksetup -setnetworkserviceenabled Thunderbolt Bridge on"))
    }

    @Test func leavesAnExistingBridgeAloneAndEnablesIt() throws {
        let sandbox = try BridgeSandbox(bridgeDefined: true)

        #expect(try sandbox.run() == 0)

        #expect(!sandbox.calls.contains { $0.hasPrefix("scutil") })
        #expect(
            sandbox.calls.contains("networksetup -setnetworkserviceenabled Thunderbolt Bridge on"))
    }

    @Test func findsARenamedServiceByItsDevice() throws {
        let renamed = BridgeSandbox.serviceOrder.replacingOccurrences(
            of: "(*) Thunderbolt Bridge", with: "(*) TB Bridge (Desk)")
        let sandbox = try BridgeSandbox(serviceOrder: renamed, bridgeDefined: false)

        #expect(try sandbox.run() == 0)

        #expect(sandbox.scutilCommands.contains("d.add Interfaces * en2 en3"))
        #expect(
            sandbox.calls.contains("networksetup -setnetworkserviceenabled TB Bridge (Desk) on"))
    }

    @Test func isPartOfTheUninstallScriptAfterLeavingTheExoLocation() {
        let script = NetworkSetupHelper.makeUninstallScript()
        let leaveLocation = script.range(of: "networksetup -switchtolocation Automatic")
        let restore = script.range(of: NetworkSetupHelper.restoreThunderboltBridgeScript)
        #expect(leaveLocation != nil && restore != nil)
        if let leaveLocation, let restore {
            #expect(leaveLocation.upperBound < restore.lowerBound)
        }
    }

    /// Any change to the daemon's script makes every Mac that has it ask for
    /// an administrator password at the next update ("EXO needs to update its
    /// system service"). Change it, and this hash, only on purpose.
    @Test func daemonScriptIsUnchanged() {
        let digest = SHA256.hash(data: Data(NetworkSetupHelper.setupScript.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        #expect(hex == "c92a331b070eb0e863054aa1ef2257487e6d052343d25da2b441da6fec2a9857")
    }
}
