import Foundation
import Testing

@testable import EXO

/// Runs the LaunchDaemon's setup script in a scratch directory. Every system
/// path in the script (including /Volumes) is redirected into that directory,
/// and the commands it runs (sleep, launchctl, networksetup, ifconfig,
/// PlistBuddy) are stubs that only record their arguments, so nothing on the
/// machine is touched.
private final class DaemonSandbox {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("EXOTests-daemon-\(UUID().uuidString)", isDirectory: true)
    var plist: URL { root.appendingPathComponent("LaunchDaemons/io.exo.networksetup.plist") }
    var script: URL { root.appendingPathComponent("Application Support/EXO/disable_bridge.sh") }
    var app: URL { root.appendingPathComponent("Applications/EXO.app", isDirectory: true) }
    /// Stands in for /Volumes. Folders created in it are plain directories on
    /// the same disk, i.e. what an unmounted disk's leftover folder looks like.
    var volumes: URL { root.appendingPathComponent("Volumes", isDirectory: true) }
    private var bin: URL { root.appendingPathComponent("bin", isDirectory: true) }
    private var callLog: URL { root.appendingPathComponent("calls.log") }

    init() throws {
        let fileManager = FileManager.default
        for directory in [
            bin, plist.deletingLastPathComponent(), script.deletingLastPathComponent(), app,
            volumes,
        ] {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        try "installed plist".write(to: plist, atomically: true, encoding: .utf8)
        try "".write(to: callLog, atomically: true, encoding: .utf8)

        for command in ["sleep", "launchctl", "networksetup", "ifconfig", "PlistBuddy"] {
            let stub = bin.appendingPathComponent(command)
            try "#!/bin/bash\necho \"\(command) $*\" >> \"\(callLog.path)\"\n"
                .write(to: stub, atomically: true, encoding: .utf8)
            try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: stub.path)
        }

        let redirected = NetworkSetupHelper.setupScript
            .replacingOccurrences(of: NetworkSetupHelper.plistDestination, with: plist.path)
            .replacingOccurrences(
                of: NetworkSetupHelper.supportDirectory,
                with: script.deletingLastPathComponent().path
            )
            .replacingOccurrences(of: "VOLUMES=\"/Volumes\"", with: "VOLUMES=\"\(volumes.path)\"")
            .replacingOccurrences(
                of: "/Library/Preferences/SystemConfiguration/preferences.plist",
                with: root.appendingPathComponent("preferences.plist").path
            )
            .replacingOccurrences(
                of: "/usr/libexec/PlistBuddy", with: bin.appendingPathComponent("PlistBuddy").path)
        let outsideSandbox = redirected.replacingOccurrences(of: root.path, with: "")
        guard !outsideSandbox.contains("/Library/"), !outsideSandbox.contains("/usr/libexec/"),
            !redirected.contains("VOLUMES=\"/Volumes\"")
        else {
            throw SandboxError.scriptTouchesSystemPaths
        }
        try redirected.write(to: script, atomically: true, encoding: .utf8)
    }

    deinit {
        try? FileManager.default.removeItem(at: root)
    }

    /// Runs the script the way launchd does (`/bin/bash <script> [app path]`)
    /// and returns its exit status.
    func runDaemon(appPath: String?) throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [script.path] + (appPath.map { [$0] } ?? [])
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

    func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    enum SandboxError: Error {
        case scriptTouchesSystemPaths
    }
}

struct NetworkSetupHelperTests {

    // MARK: - The daemon stops when the app is gone (#2279)

    @Test func daemonRemovesItselfAndLeavesTheNetworkAloneWhenTheAppIsGone() throws {
        let sandbox = try DaemonSandbox()
        try FileManager.default.removeItem(at: sandbox.app)  // dragged to the Trash

        #expect(try sandbox.runDaemon(appPath: sandbox.app.path) == 0)

        #expect(!sandbox.exists(sandbox.plist))
        #expect(!sandbox.exists(sandbox.script))
        #expect(!sandbox.exists(sandbox.script.deletingLastPathComponent()))
        #expect(sandbox.calls.contains("launchctl bootout system/io.exo.networksetup"))
        #expect(
            !sandbox.calls.contains {
                $0.hasPrefix("networksetup") || $0.hasPrefix("ifconfig")
                    || $0.hasPrefix("PlistBuddy")
            })
    }

    @Test func daemonKeepsRunningWhenTheAppsDiskIsNotMounted() throws {
        // External disks mount at login (or may be unplugged), so at boot the
        // app's folder doesn't exist at all.
        let sandbox = try DaemonSandbox()
        let app = sandbox.volumes.appendingPathComponent("External/Applications/EXO.app").path

        #expect(try sandbox.runDaemon(appPath: app) == 0)

        #expect(sandbox.exists(sandbox.plist))
        #expect(sandbox.calls.contains("networksetup -switchtolocation exo"))
        #expect(!sandbox.calls.contains { $0.hasPrefix("launchctl") })
    }

    @Test func daemonKeepsRunningWhenAnUnmountedDiskLeftItsFolderBehind() throws {
        // App at the root of a disk that isn't mounted, with an empty
        // /Volumes/<name> folder left over from an earlier mount.
        let sandbox = try DaemonSandbox()
        let volume = sandbox.volumes.appendingPathComponent("External", isDirectory: true)
        try FileManager.default.createDirectory(at: volume, withIntermediateDirectories: true)

        #expect(try sandbox.runDaemon(appPath: volume.appendingPathComponent("EXO.app").path) == 0)

        #expect(sandbox.exists(sandbox.plist))
        #expect(sandbox.calls.contains("networksetup -switchtolocation exo"))
        #expect(!sandbox.calls.contains { $0.hasPrefix("launchctl") })
    }

    @Test func daemonConfiguresTheNetworkWhileTheAppIsInstalled() throws {
        let sandbox = try DaemonSandbox()

        #expect(try sandbox.runDaemon(appPath: sandbox.app.path) == 0)

        #expect(sandbox.exists(sandbox.plist))
        #expect(sandbox.exists(sandbox.script))
        #expect(sandbox.calls.contains("networksetup -switchtolocation exo"))
        #expect(!sandbox.calls.contains { $0.hasPrefix("launchctl") })
    }

    @Test func daemonInstalledWithoutAnAppPathKeepsConfiguringTheNetwork() throws {
        // Daemons installed by older versions, or from a translocated app,
        // have no app path to check.
        let sandbox = try DaemonSandbox()

        #expect(try sandbox.runDaemon(appPath: nil) == 0)

        #expect(sandbox.exists(sandbox.plist))
        #expect(sandbox.calls.contains("networksetup -switchtolocation exo"))
        #expect(!sandbox.calls.contains { $0.hasPrefix("launchctl") })
    }

    // MARK: - Which app path the daemon is given

    @Test func installedAppPathIsPassedToTheDaemon() {
        #expect(
            NetworkSetupHelper.daemonAppPath(forBundlePath: "/Applications/EXO.app")
                == "/Applications/EXO.app")
        #expect(
            NetworkSetupHelper.daemonProgramArguments(appPath: "/Applications/EXO.app")
                == ["/bin/bash", NetworkSetupHelper.scriptDestination, "/Applications/EXO.app"])
    }

    @Test func translocatedAppPathIsNotPassedToTheDaemon() {
        let translocated =
            "/private/var/folders/xy/abc123/T/AppTranslocation/6F1C2D3E-0000-4000-8000-000000000000/d/EXO.app"

        #expect(NetworkSetupHelper.daemonAppPath(forBundlePath: translocated) == nil)
        #expect(
            NetworkSetupHelper.daemonProgramArguments(appPath: nil)
                == ["/bin/bash", NetworkSetupHelper.scriptDestination])
    }

    @Test func installedDaemonForThisAppOrAnotherExistingCopyIsKept() {
        let installed = NetworkSetupHelper.daemonProgramArguments(appPath: "/Applications/EXO.app")
        let noOtherCopies: (String) -> Bool = { _ in false }
        let applicationsCopyExists: (String) -> Bool = { $0 == "/Applications/EXO.app" }

        #expect(
            NetworkSetupHelper.installedDaemonMatches(
                programArguments: installed, appPath: "/Applications/EXO.app",
                appExists: noOtherCopies))
        // A second copy (e.g. a development build) while /Applications/EXO.app
        // is still there: no reinstall, so no password prompt on every switch.
        #expect(
            NetworkSetupHelper.installedDaemonMatches(
                programArguments: installed, appPath: "/Users/someone/Build/EXO.app",
                appExists: applicationsCopyExists))
    }

    @Test func installedDaemonIsReplacedWhenItsAppIsGoneOrItHasNoAppPath() {
        let installed = NetworkSetupHelper.daemonProgramArguments(appPath: "/Applications/EXO.app")
        let legacy = ["/bin/bash", NetworkSetupHelper.scriptDestination]
        let nothingExists: (String) -> Bool = { _ in false }

        // The app was moved: the recorded path no longer exists.
        #expect(
            !NetworkSetupHelper.installedDaemonMatches(
                programArguments: installed, appPath: "/Users/someone/Applications/EXO.app",
                appExists: nothingExists))
        #expect(
            !NetworkSetupHelper.installedDaemonMatches(
                programArguments: legacy, appPath: "/Applications/EXO.app",
                appExists: nothingExists))
        #expect(
            !NetworkSetupHelper.installedDaemonMatches(
                programArguments: nil, appPath: "/Applications/EXO.app",
                appExists: nothingExists))
        #expect(
            !NetworkSetupHelper.installedDaemonMatches(
                programArguments: ["/bin/bash", "/tmp/other.sh", "/Applications/EXO.app"],
                appPath: "/Applications/EXO.app", appExists: nothingExists))
    }

    @Test func translocatedAppAcceptsAnyDaemonRunningTheCurrentScript() {
        let installed = NetworkSetupHelper.daemonProgramArguments(appPath: "/Applications/EXO.app")
        let legacy = ["/bin/bash", NetworkSetupHelper.scriptDestination]
        let nothingExists: (String) -> Bool = { _ in false }

        #expect(
            NetworkSetupHelper.installedDaemonMatches(
                programArguments: installed, appPath: nil, appExists: nothingExists))
        #expect(
            NetworkSetupHelper.installedDaemonMatches(
                programArguments: legacy, appPath: nil, appExists: nothingExists))
        #expect(
            !NetworkSetupHelper.installedDaemonMatches(
                programArguments: ["/bin/bash", "/tmp/other.sh"], appPath: nil,
                appExists: nothingExists))
    }

    @Test func appPathIsEscapedForTheLaunchDaemonPlist() {
        #expect(
            NetworkSetupHelper.xmlEscaped("/Volumes/R&D <\"tools\">/Bob's EXO.app")
                == "/Volumes/R&amp;D &lt;&quot;tools&quot;&gt;/Bob&apos;s EXO.app")

        let installer = NetworkSetupHelper.makeInstallerScript(appPath: "/Volumes/R&D/EXO.app")
        #expect(installer.contains("<string>/Volumes/R&amp;D/EXO.app</string>"))
    }
}
