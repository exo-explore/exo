import AppKit
import Foundation
import os.log

enum NetworkSetupHelper {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "NetworkSetup")
    private static let daemonLabel = "io.exo.networksetup"
    private static let scriptDestination =
        "/Library/Application Support/EXO/disable_bridge_enable_dhcp.sh"
    private static let plistDestination = "/Library/LaunchDaemons/io.exo.networksetup.plist"
    private static let requiredStartInterval: Int = 1791

    private static let setupScript = """
        #!/usr/bin/env bash

        set -euo pipefail

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

    static func ensureLaunchDaemonInstalled() {
        Task.detached {
            do {
                if daemonAlreadyInstalled() {
                    return
                }
                try await installLaunchDaemon()
                logger.info("Network setup launch daemon installed and started")
            } catch {
                logger.error(
                    "Network setup launch daemon failed: \(error.localizedDescription, privacy: .public)"
                )
            }
        }
    }

    private static func daemonAlreadyInstalled() -> Bool {
        let manager = FileManager.default
        let scriptExists = manager.fileExists(atPath: scriptDestination)
        let plistExists = manager.fileExists(atPath: plistDestination)
        guard scriptExists, plistExists else { return false }
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

    private static func installLaunchDaemon() async throws {
        let installerScript = makeInstallerScript()
        try runShellAsAdmin(installerScript)
    }

    private static func makeInstallerScript() -> String {
        """
        set -euo pipefail

        LABEL="\(daemonLabel)"
        SCRIPT_DEST="\(scriptDestination)"
        PLIST_DEST="\(plistDestination)"

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

        launchctl bootout system/"$LABEL" >/dev/null 2>&1 || true
        launchctl bootstrap system "$PLIST_DEST"
        launchctl enable system/"$LABEL"
        launchctl kickstart -k system/"$LABEL"
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
