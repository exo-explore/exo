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
