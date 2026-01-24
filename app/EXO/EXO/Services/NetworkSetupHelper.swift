import AppKit
import Foundation
import os.log

enum NetworkSetupHelper {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "NetworkSetup")
    private static let daemonLabel = "io.exo.networksetup"
    private static let scriptDestination =
        "/Library/Application Support/EXO/disable_bridge.sh"
    // Legacy script path from older versions
    private static let legacyScriptDestination =
        "/Library/Application Support/EXO/disable_bridge_enable_dhcp.sh"
    private static let plistDestination = "/Library/LaunchDaemons/io.exo.networksetup.plist"
    private static let requiredStartInterval: Int = 1786

    private static let setupScript = """
        #!/usr/bin/env bash

        set -euo pipefail

        # Wait for macOS to finish network setup after boot
        sleep 30

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
                    "EXO needs to install a system service to automatically disable Thunderbolt Bridge on startup. This prevents network loops when connecting multiple Macs via Thunderbolt.\n\nYou will be prompted for your administrator password."
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
        if let programArgs = plist["ProgramArguments"] as? [String],
            programArgs.contains(scriptDestination) == false
        {
            return false
        }
        return true
    }

    private static func installLaunchDaemon() throws {
        let installerScript = makeInstallerScript()
        try runShellAsAdmin(installerScript)
    }

    private static func makeInstallerScript() -> String {
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
            <string>/bin/bash</string>
            <string>\(scriptDestination)</string>
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
        networksetup -switchtolocation Automatic 2>/dev/null || true

        # Delete the exo network location if it exists
        networksetup -listlocations | grep -q '^exo$' && {
          networksetup -deletelocation exo 2>/dev/null || true
        } || true

        # Re-enable any Thunderbolt Bridge service if it exists
        # We find it dynamically by looking for bridges containing Thunderbolt interfaces
        find_and_enable_thunderbolt_bridge() {
          # Get Thunderbolt interface devices from hardware ports
          tb_devices=$(networksetup -listallhardwareports 2>/dev/null | awk '
            /^Hardware Port:/ { port = tolower(substr($0, 16)) }
            /^Device:/ { if (port ~ /thunderbolt/) print substr($0, 9) }
          ')
          [ -z "$tb_devices" ] && return 0

          # For each bridge device, check if it contains Thunderbolt interfaces
          for bridge in bridge0 bridge1 bridge2; do
            members=$(ifconfig "$bridge" 2>/dev/null | awk '/member:/ {print $2}')
            [ -z "$members" ] && continue

            for tb_dev in $tb_devices; do
              if echo "$members" | grep -qx "$tb_dev"; then
                # Find the service name for this bridge device
                service_name=$(networksetup -listnetworkserviceorder 2>/dev/null | awk -v dev="$bridge" '
                  /^\\([0-9*]/ { gsub(/^\\([0-9*]+\\) /, ""); svc = $0 }
                  /Device:/ && $0 ~ dev { print svc; exit }
                ')
                if [ -n "$service_name" ]; then
                  networksetup -setnetworkserviceenabled "$service_name" on 2>/dev/null || true
                  return 0
                fi
              fi
            done
          done
        }
        find_and_enable_thunderbolt_bridge

        echo "EXO network components removed successfully"
        """
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
