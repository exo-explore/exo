import AppKit
import Foundation
import os.log

enum NetworkSetupHelper {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "NetworkSetup")
    private static let daemonLabel = "io.exo.networksetup"
    static let scriptDestination =
        "/Library/Application Support/EXO/disable_bridge.sh"
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
        if [[ -n "$EXO_APP_PATH" && ! -d "$EXO_APP_PATH" ]]; then
          echo "EXO.app not found at $EXO_APP_PATH; removing the \(daemonLabel) LaunchDaemon"
          rm -f "\(plistDestination)" "$0"
          rmdir "$(dirname "$0")" 2>/dev/null || true
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

    /// Prompts user and installs the LaunchDaemon if not already installed.
    /// Shows an alert explaining what will be installed before requesting admin privileges.
    static func promptAndInstallIfNeeded() {
        // Use .utility priority to match NSAppleScript's internal QoS and avoid priority inversion
        Task.detached(priority: .utility) {
            // If already correctly installed, skip
            if daemonAlreadyInstalled() {
                return
            }

            // Show alert on main thread
            let shouldInstall = await MainActor.run {
                let alert = NSAlert()
                alert.messageText = "EXO Network Configuration"
                alert.informativeText =
                    "EXO needs to install a system service to configure local networking. This will disable Thunderbolt Bridge (preventing packet storms) and install a Network Location.\n\nYou will be prompted for your password."
                alert.alertStyle = .informational
                alert.addButton(withTitle: "Install")
                alert.addButton(withTitle: "Not Now")
                return alert.runModal() == .alertFirstButtonReturn
            }

            guard shouldInstall else {
                logger.info("User deferred network setup daemon installation")
                return
            }

            do {
                try installLaunchDaemon()
                logger.info("Network setup launch daemon installed and started")
            } catch {
                logger.error(
                    "Network setup launch daemon failed: \(error.localizedDescription, privacy: .public)"
                )
            }
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
            appPath: currentAppPath()
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

    /// Whether an installed daemon's ProgramArguments are what this copy of
    /// the app would install. When the app's path isn't known (translocated),
    /// any daemon running the current script is accepted rather than asking
    /// to reinstall on every launch.
    static func installedDaemonMatches(programArguments: [String]?, appPath: String?) -> Bool {
        guard let programArguments else { return false }
        if let appPath {
            return programArguments == daemonProgramArguments(appPath: appPath)
        }
        return Array(programArguments.prefix(2)) == daemonProgramArguments(appPath: nil)
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

    private static func makeUninstallScript() -> String {
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

        # Re-enable any Thunderbolt Bridge service if it exists
        # We find it dynamically by looking for bridges containing Thunderbolt interfaces
        find_and_enable_thunderbolt_bridge() {
          # Get Thunderbolt interface devices from hardware ports
          tb_devices=$(networksetup -listallhardwareports 2>/dev/null | awk '
            /^Hardware Port:/ { port = tolower(substr($0, 16)) }
            /^Device:/ { if (port ~ /thunderbolt/) print substr($0, 9) }
          ') || true
          [ -z "$tb_devices" ] && return 0

          # For each bridge device, check if it contains Thunderbolt interfaces
          for bridge in bridge0 bridge1 bridge2; do
            members=$(ifconfig "$bridge" 2>/dev/null | awk '/member:/ {print $2}') || true
            [ -z "$members" ] && continue

            for tb_dev in $tb_devices; do
              if echo "$members" | grep -qx "$tb_dev"; then
                # Find the service name for this bridge device
                service_name=$(networksetup -listnetworkserviceorder 2>/dev/null | awk -v dev="$bridge" '
                  /^\\([0-9*]/ { gsub(/^\\([0-9*]+\\) /, ""); svc = $0 }
                  /Device:/ && $0 ~ dev { print svc; exit }
                ') || true
                if [ -n "$service_name" ]; then
                  networksetup -setnetworkserviceenabled "$service_name" on 2>/dev/null || true
                  return 0
                fi
              fi
            done
          done
          return 0
        }
        find_and_enable_thunderbolt_bridge || true

        echo "EXO network components removed successfully"
        """
    }

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

    private static func runShellAsAdmin(_ script: String) throws {
        let escapedScript =
            script
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")

        let appleScriptSource = """
            do shell script "\(escapedScript)" with administrator privileges
            """

        guard let appleScript = NSAppleScript(source: appleScriptSource) else {
            throw NetworkSetupError.scriptCreationFailed
        }

        var errorInfo: NSDictionary?
        appleScript.executeAndReturnError(&errorInfo)

        if let errorInfo {
            let message = errorInfo[NSAppleScript.errorMessage] as? String ?? "Unknown error"
            throw NetworkSetupError.executionFailed(message)
        }
    }
}

enum NetworkSetupError: LocalizedError {
    case scriptCreationFailed
    case executionFailed(String)

    var errorDescription: String? {
        switch self {
        case .scriptCreationFailed:
            return "Failed to create AppleScript for network setup"
        case .executionFailed(let message):
            return "Network setup script failed: \(message)"
        }
    }
}
