import AppKit
import Foundation
import os.log

enum NetworkSetupHelper {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "NetworkSetup")
    private static let daemonLabel = "io.exo.networksetup"
    static let supportDirectory = "/Library/Application Support/EXO"
    static let scriptDestination = "\(supportDirectory)/disable_bridge.sh"
    // Legacy script path from older versions
    private static let legacyScriptDestination =
        "/Library/Application Support/EXO/disable_bridge_enable_dhcp.sh"
    static let plistDestination = "/Library/LaunchDaemons/io.exo.networksetup.plist"
    private static let requiredStartInterval: Int = 1786

    static let setupScript = """
        #!/usr/bin/env bash

        set -euo pipefail

        # Wait for macOS to finish network setup after boot
        sleep 20

        # The daemon is installed with EXO.app's path as its first argument. If
        # the app has since been deleted (e.g. dragged to the Trash), stop
        # changing the network and remove this daemon. Settings it already
        # applied are left as they are; uninstall-exo.sh restores them.
        EXO_APP_PATH="${1:-}"
        VOLUMES="/Volumes"
        exo_app_deleted() {
          [[ -n "$EXO_APP_PATH" && ! -e "$EXO_APP_PATH" ]] || return 1
          # If the folder that held the app is missing too, the disk it is on is
          # most likely just not mounted (external disks mount at login, and can
          # be unplugged), so the app may well still exist.
          [[ -d "$(dirname "$EXO_APP_PATH")" ]] || return 1
          # An unmounted disk can leave an empty /Volumes/<name> folder behind,
          # so for an app on another disk also check that the disk is mounted.
          if [[ "$EXO_APP_PATH" == "$VOLUMES"/*/* ]]; then
            local volume="${EXO_APP_PATH#"$VOLUMES"/}"
            volume="$VOLUMES/${volume%%/*}"
            [[ "$(stat -f %d "$volume")" != "$(stat -f %d "$VOLUMES")" ]] || return 1
          fi
        }
        if exo_app_deleted; then
          echo "EXO.app not found at $EXO_APP_PATH; removing the \(daemonLabel) LaunchDaemon"
          rm -f "\(plistDestination)" "\(scriptDestination)"
          rmdir "\(supportDirectory)" 2>/dev/null || true
          launchctl bootout system/\(daemonLabel) 2>/dev/null || true
          exit 0
        fi

        PREFS="/Library/Preferences/SystemConfiguration/preferences.plist"

        # Remove bridge0 interface
        ifconfig bridge0 &>/dev/null && {
          ifconfig bridge0 | grep -q 'member' && {
            ifconfig bridge0 | awk '/member/ {print $2}' | xargs -n1 ifconfig bridge0 deletem 2>/dev/null || true
          }
          ifconfig bridge0 destroy 2>/dev/null || true
        }

        # Remove Thunderbolt Bridge from VirtualNetworkInterfaces in preferences.plist
        /usr/libexec/PlistBuddy -c "Delete :VirtualNetworkInterfaces:Bridge:bridge0" "$PREFS" 2>/dev/null || true

        networksetup -listlocations | grep -q exo || {
          networksetup -createlocation exo
        }

        networksetup -switchtolocation exo
        networksetup -listallhardwareports \\
          | awk -F': ' '/Hardware Port: / {print $2}' \\
          | while IFS=":" read -r name; do
              case "$name" in
                "Ethernet Adapter"*)
                        ;;
                "Thunderbolt Bridge")
                        ;;
                "Thunderbolt "*)
                  networksetup -listallnetworkservices \\
                    | grep -q "EXO $name" \\
                      || networksetup -createnetworkservice "EXO $name" "$name" 2>/dev/null \\
                      || continue
                  networksetup -setdhcp "EXO $name"
                        ;;
                *)
                  networksetup -listallnetworkservices \\
                    | grep -q "$name" \\
                      || networksetup -createnetworkservice "$name" "$name" 2>/dev/null \\
                      || continue
                        ;;
              esac
            done

        networksetup -listnetworkservices | grep -q "Thunderbolt Bridge" && {
          networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off
        } || true
        """

    /// Whether this Mac has EXO's Thunderbolt networking service, and whether
    /// it is the current version.
    enum SetupState: Equatable {
        case notInstalled
        /// Installed by an older EXO. It keeps working, but should be
        /// replaced.
        case outdated
        case current
    }

    /// Reads the installed service's files: cheap enough to check when a menu
    /// opens.
    static var setupState: SetupState {
        if daemonAlreadyInstalled() { return .current }
        return hasInstalledComponents() ? .outdated : .notInstalled
    }

    /// Whether the service is missing or out of date. It only matters for
    /// Macs connected to each other by Thunderbolt, so it is offered in that
    /// context (or from Settings), never when the app starts.
    static var needsSetup: Bool {
        setupState != .current
    }

    static let setupDeclinedKey = "EXONetworkSetupDeclined"

    /// Whether the user chose Not Now when the service was offered. The app
    /// then stops offering it on its own; Settings still can.
    static func setupWasDeclined(defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: setupDeclinedKey)
    }

    static func recordSetupDeclined(_ declined: Bool, defaults: UserDefaults = .standard) {
        defaults.set(declined, forKey: setupDeclinedKey)
    }

    enum SetupOutcome: Equatable {
        case installed
        /// The user cancelled the administrator password prompt.
        case cancelled
    }

    /// Installs (or updates) the service; macOS asks for an administrator
    /// password. Throws if the installer itself fails.
    static func setUp() async throws -> SetupOutcome {
        // NSAppleScript's administrator prompt blocks, and its script runs at
        // .utility; matching that avoids a priority inversion.
        try await Task.detached(priority: .utility) {
            do {
                try installLaunchDaemon()
                logger.info("Network setup launch daemon installed and started")
                return SetupOutcome.installed
            } catch NetworkSetupError.cancelled {
                logger.info("Network setup cancelled at the password prompt")
                return SetupOutcome.cancelled
            }
        }.value
    }

    /// Says what setting up Thunderbolt networking changes, then sets it up
    /// if the user agrees. "Not Now" is remembered. A cancelled password
    /// prompt does nothing; a real failure is shown.
    @MainActor
    static func offerSetUp() async {
        let isUpdate = setupState == .outdated
        let alert = NSAlert()
        alert.messageText =
            isUpdate ? "Update Thunderbolt Networking?" : "Set Up Thunderbolt Networking?"
        alert.informativeText =
            isUpdate
            ? "EXO's network service needs an update. It keeps Thunderbolt Bridge off as before, and now removes itself if EXO is deleted. You'll be asked for an administrator password."
            : "Macs connected by Thunderbolt can then talk to each other directly. EXO installs a small system service that turns off Thunderbolt Bridge, gives each Thunderbolt port its own connection, and keeps this Mac on a network location named “exo”. You'll be asked for an administrator password."
        alert.addButton(withTitle: isUpdate ? "Update" : "Set Up")
        alert.addButton(withTitle: "Not Now")
        guard alert.runModalInFront() == .alertFirstButtonReturn else {
            recordSetupDeclined(true)
            return
        }
        recordSetupDeclined(false)
        do {
            _ = try await setUp()
        } catch {
            logger.error(
                "Network setup launch daemon failed: \(error.localizedDescription, privacy: .public)"
            )
            let failure = NSAlert()
            failure.messageText = "Couldn't Set Up Thunderbolt Networking"
            failure.informativeText = error.localizedDescription
            failure.alertStyle = .warning
            failure.addButton(withTitle: "OK")
            failure.runModalInFront()
        }
    }

    /// Removes all EXO network setup components from the system.
    /// This includes the LaunchDaemon, scripts, logs, and network location.
    /// Requires admin privileges.
    static func uninstall() throws {
        let uninstallScript = makeUninstallScript()
        try runShellAsAdmin(uninstallScript)
        logger.info("EXO network setup components removed successfully")
    }

    /// Checks if there are any EXO network components installed that need cleanup
    static func hasInstalledComponents() -> Bool {
        let manager = FileManager.default
        let scriptExists = manager.fileExists(atPath: scriptDestination)
        let legacyScriptExists = manager.fileExists(atPath: legacyScriptDestination)
        let plistExists = manager.fileExists(atPath: plistDestination)
        return scriptExists || legacyScriptExists || plistExists
    }

    private static func daemonAlreadyInstalled() -> Bool {
        let manager = FileManager.default
        let scriptExists = manager.fileExists(atPath: scriptDestination)
        let plistExists = manager.fileExists(atPath: plistDestination)
        guard scriptExists, plistExists else { return false }
        guard
            let installedScript = try? String(contentsOfFile: scriptDestination, encoding: .utf8),
            installedScript.trimmingCharacters(in: .whitespacesAndNewlines)
                == setupScript.trimmingCharacters(in: .whitespacesAndNewlines)
        else {
            return false
        }
        guard
            let data = try? Data(contentsOf: URL(fileURLWithPath: plistDestination)),
            let plist = try? PropertyListSerialization.propertyList(
                from: data, options: [], format: nil) as? [String: Any]
        else {
            return false
        }
        guard
            let interval = plist["StartInterval"] as? Int,
            interval == requiredStartInterval
        else {
            return false
        }
        return installedDaemonMatches(
            programArguments: plist["ProgramArguments"] as? [String],
            appPath: currentAppPath(),
            appExists: { manager.fileExists(atPath: $0) }
        )
    }

    /// The path of the running EXO.app to hand to the daemon, or nil if it
    /// isn't a stable location. A quarantined app opened straight from
    /// Downloads or a disk image runs from a randomized App Translocation
    /// path that disappears when the app quits, which would make the daemon
    /// remove itself.
    static func daemonAppPath(forBundlePath bundlePath: String) -> String? {
        bundlePath.contains("/AppTranslocation/") ? nil : bundlePath
    }

    private static func currentAppPath() -> String? {
        daemonAppPath(forBundlePath: Bundle.main.bundlePath)
    }

    /// The LaunchDaemon's ProgramArguments: the setup script, followed by the
    /// app's path when it is known so the script can tell if the app is gone.
    static func daemonProgramArguments(appPath: String?) -> [String] {
        ["/bin/bash", scriptDestination] + (appPath.map { [$0] } ?? [])
    }

    /// Whether an installed daemon can be kept as it is. It must run the
    /// current script. If it was given an app path, that path must be this
    /// app or another copy that still exists, so switching between two copies
    /// (e.g. a development build and /Applications) doesn't ask to reinstall
    /// every time. A daemon without an app path is replaced so it can learn
    /// where the app is, unless this app's own path isn't known (translocated).
    static func installedDaemonMatches(
        programArguments: [String]?,
        appPath: String?,
        appExists: (String) -> Bool
    ) -> Bool {
        guard let programArguments,
            Array(programArguments.prefix(2)) == daemonProgramArguments(appPath: nil)
        else {
            return false
        }
        guard let appPath else { return true }
        guard programArguments.count == 3 else { return false }
        let recordedAppPath = programArguments[2]
        return recordedAppPath == appPath || appExists(recordedAppPath)
    }

    /// Escapes text for use inside a property list `<string>` element.
    static func xmlEscaped(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&apos;")
    }

    private static func installLaunchDaemon() throws {
        let installerScript = makeInstallerScript(appPath: currentAppPath())
        try runShellAsAdmin(installerScript)
    }

    static func makeInstallerScript(appPath: String?) -> String {
        """
        set -euo pipefail

        LABEL="\(daemonLabel)"
        SCRIPT_DEST="\(scriptDestination)"
        LEGACY_SCRIPT_DEST="\(legacyScriptDestination)"
        PLIST_DEST="\(plistDestination)"
        LOG_OUT="/var/log/\(daemonLabel).log"
        LOG_ERR="/var/log/\(daemonLabel).err.log"

        # First, completely remove any existing installation
        launchctl bootout system/"$LABEL" 2>/dev/null || true
        rm -f "$PLIST_DEST"
        rm -f "$SCRIPT_DEST"
        rm -f "$LEGACY_SCRIPT_DEST"
        rm -f "$LOG_OUT" "$LOG_ERR"

        # Install fresh
        mkdir -p "$(dirname "$SCRIPT_DEST")"

        cat > "$SCRIPT_DEST" <<'EOF_SCRIPT'
        \(setupScript)
        EOF_SCRIPT
        chmod 755 "$SCRIPT_DEST"

        cat > "$PLIST_DEST" <<'EOF_PLIST'
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
          <key>Label</key>
          <string>\(daemonLabel)</string>
          <key>ProgramArguments</key>
          <array>
        \(programArgumentsXML(appPath: appPath))
          </array>
          <key>StartInterval</key>
          <integer>\(requiredStartInterval)</integer>
          <key>RunAtLoad</key>
          <true/>
          <key>StandardOutPath</key>
          <string>/var/log/\(daemonLabel).log</string>
          <key>StandardErrorPath</key>
          <string>/var/log/\(daemonLabel).err.log</string>
        </dict>
        </plist>
        EOF_PLIST

        launchctl bootstrap system "$PLIST_DEST"
        launchctl enable system/"$LABEL"
        launchctl kickstart -k system/"$LABEL"
        """
    }

    private static func programArgumentsXML(appPath: String?) -> String {
        daemonProgramArguments(appPath: appPath)
            .map { "    <string>\(xmlEscaped($0))</string>" }
            .joined(separator: "\n")
    }

    static func makeUninstallScript() -> String {
        """
        set -euo pipefail

        LABEL="\(daemonLabel)"
        SCRIPT_DEST="\(scriptDestination)"
        LEGACY_SCRIPT_DEST="\(legacyScriptDestination)"
        PLIST_DEST="\(plistDestination)"
        LOG_OUT="/var/log/\(daemonLabel).log"
        LOG_ERR="/var/log/\(daemonLabel).err.log"

        # Unload the LaunchDaemon if running
        launchctl bootout system/"$LABEL" 2>/dev/null || true

        # Remove LaunchDaemon plist
        rm -f "$PLIST_DEST"

        # Remove the script (current and legacy paths) and parent directory if empty
        rm -f "$SCRIPT_DEST"
        rm -f "$LEGACY_SCRIPT_DEST"
        rmdir "$(dirname "$SCRIPT_DEST")" 2>/dev/null || true

        # Remove log files
        rm -f "$LOG_OUT" "$LOG_ERR"

        # Switch back to Automatic network location
        networksetup -switchtolocation Automatic >/dev/null 2>&1 || true

        # Delete the exo network location if it exists
        networksetup -listlocations 2>/dev/null | grep -q '^exo$' && {
          networksetup -deletelocation exo >/dev/null 2>&1 || true
        } || true

        \(restoreThunderboltBridgeScript)

        echo "EXO network components removed successfully"
        """
    }

    /// Puts Thunderbolt Bridge back once the Mac is on Automatic again; part
    /// of the uninstall script. macOS bridges the Mac's Thunderbolt ports as
    /// bridge0, and the setup script deletes that interface (not just the
    /// service), so the "Thunderbolt Bridge" service is left with no
    /// interface. This recreates the interface the way macOS defines it (the
    /// Thunderbolt ports, with `__AUTO__ = thunderbolt-bridge`) and enables
    /// the service. A failure here doesn't stop the rest of the uninstall.
    static let restoreThunderboltBridgeScript = """
        BRIDGE_DEVICE="bridge0"
        NETWORK_PREFS="/Library/Preferences/SystemConfiguration/preferences.plist"

        # The service in the current location that uses bridge0, found by
        # device so a renamed or localised service is found too.
        bridge_service() {
          networksetup -listnetworkserviceorder 2>/dev/null | awk -v device="Device: $BRIDGE_DEVICE)" '
            /^\\([0-9*]+\\) / { service = $0; sub(/^\\([0-9*]+\\) /, "", service) }
            index($0, device) { print service; exit }
          '
        }

        # The Mac's Thunderbolt ports (en2, en3, ...), not counting bridges.
        thunderbolt_ports() {
          networksetup -listallhardwareports 2>/dev/null | awk '
            /^Hardware Port:/ { port = tolower(substr($0, 16)) }
            /^Device:/ { device = substr($0, 9); if (port ~ /thunderbolt/ && device !~ /^bridge/) print device }
          '
        }

        TB_BRIDGE_SERVICE="$(bridge_service)" || TB_BRIDGE_SERVICE=""
        if [ -n "$TB_BRIDGE_SERVICE" ] &&
          ! /usr/libexec/PlistBuddy -c "Print :VirtualNetworkInterfaces:Bridge:$BRIDGE_DEVICE" "$NETWORK_PREFS" >/dev/null 2>&1; then
          TB_PORTS="$(thunderbolt_ports | tr '\\n' ' ')" || TB_PORTS=""
          if [ -n "$(printf '%s' "$TB_PORTS" | tr -d ' ')" ]; then
            # Leaving the exo location leaves the Thunderbolt ports down, and
            # a port that is down can't join a bridge.
            for port in $TB_PORTS; do
              ifconfig "$port" up 2>/dev/null || true
            done
            scutil --prefs >/dev/null <<EOF_BRIDGE || true
        d.init
        d.add Interfaces * $TB_PORTS
        set /VirtualNetworkInterfaces/Bridge/$BRIDGE_DEVICE
        d.init
        d.add __AUTO__ thunderbolt-bridge
        set /VirtualNetworkInterfaces/Bridge/$BRIDGE_DEVICE/Options
        commit
        apply
        quit
        EOF_BRIDGE
          fi
        fi
        # Disabled services are listed with a leading "*".
        if [ -z "$TB_BRIDGE_SERVICE" ] &&
          networksetup -listallnetworkservices 2>/dev/null | sed 's/^\\*//' | grep -x "Thunderbolt Bridge" >/dev/null; then
          TB_BRIDGE_SERVICE="Thunderbolt Bridge"
        fi
        if [ -n "$TB_BRIDGE_SERVICE" ]; then
          networksetup -setnetworkserviceenabled "$TB_BRIDGE_SERVICE" on 2>/dev/null || true
        fi
        """

    /// Direct install without GUI (requires root).
    /// Returns true on success, false on failure.
    static func installDirectly() -> Bool {
        let script = makeInstallerScript(appPath: currentAppPath())
        return runShellDirectly(script)
    }

    /// Direct uninstall without GUI (requires root).
    /// Returns true on success, false on failure.
    static func uninstallDirectly() -> Bool {
        let script = makeUninstallScript()
        return runShellDirectly(script)
    }

    /// Run a shell script directly via Process (no AppleScript, requires root).
    /// Returns true on success, false on failure.
    private static func runShellDirectly(_ script: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", script]

        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        do {
            try process.run()
            process.waitUntilExit()

            let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()

            if let output = String(data: outputData, encoding: .utf8), !output.isEmpty {
                print(output)
            }
            if let errorOutput = String(data: errorData, encoding: .utf8), !errorOutput.isEmpty {
                fputs(errorOutput, stderr)
            }

            if process.terminationStatus == 0 {
                logger.info("Shell script completed successfully")
                return true
            } else {
                logger.error("Shell script failed with exit code \(process.terminationStatus)")
                return false
            }
        } catch {
            logger.error(
                "Failed to run shell script: \(error.localizedDescription, privacy: .public)")
            fputs("Error: \(error.localizedDescription)\n", stderr)
            return false
        }
    }

    /// AppleScript that runs `script` with `do shell script`, which hands it
    /// to /bin/sh.
    static func appleScriptSource(running script: String, asAdministrator: Bool) -> String {
        let escapedScript =
            script
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "do shell script \"\(escapedScript)\""
            + (asAdministrator ? " with administrator privileges" : "")
    }

    private static func runShellAsAdmin(_ script: String) throws {
        let appleScriptSource = appleScriptSource(running: script, asAdministrator: true)

        guard let appleScript = NSAppleScript(source: appleScriptSource) else {
            throw NetworkSetupError.scriptCreationFailed
        }

        var errorInfo: NSDictionary?
        appleScript.executeAndReturnError(&errorInfo)

        if let errorInfo {
            throw NetworkSetupError(
                appleScriptErrorNumber: errorInfo[NSAppleScript.errorNumber] as? Int,
                message: errorInfo[NSAppleScript.errorMessage] as? String)
        }
    }
}

enum NetworkSetupError: LocalizedError, Equatable {
    case scriptCreationFailed
    /// The user cancelled the administrator password prompt. Not a failure:
    /// callers should do nothing.
    case cancelled
    case executionFailed(String)

    /// AppleScript's "User canceled." (`userCanceledErr`).
    static let userCancelledErrorNumber = -128

    init(appleScriptErrorNumber: Int?, message: String?) {
        if appleScriptErrorNumber == Self.userCancelledErrorNumber {
            self = .cancelled
        } else {
            self = .executionFailed(message ?? "Unknown error")
        }
    }

    var errorDescription: String? {
        switch self {
        case .scriptCreationFailed:
            return "Failed to create AppleScript for network setup"
        case .cancelled:
            return "The administrator password prompt was cancelled."
        case .executionFailed(let message):
            return "Network setup script failed: \(message)"
        }
    }
}
