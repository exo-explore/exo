import Foundation
import os.log

/// Utility for dynamically detecting Thunderbolt Bridge network services.
/// This mirrors the Python logic in info_gatherer.py - we never assume the service
/// is named "Thunderbolt Bridge", instead we find bridges containing Thunderbolt interfaces.
enum ThunderboltBridgeDetector {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "ThunderboltBridgeDetector")

    struct CommandResult {
        let exitCode: Int32
        let output: String
        let error: String
    }

    /// Find the network service name of a bridge containing Thunderbolt interfaces.
    /// Returns nil if no such bridge exists.
    static func findThunderboltBridgeServiceName() -> String? {
        // 1. Get all Thunderbolt interface devices (e.g., en2, en3)
        guard let thunderboltDevices = getThunderboltDevices(), !thunderboltDevices.isEmpty else {
            logger.debug("No Thunderbolt devices found")
            return nil
        }
        logger.debug("Found Thunderbolt devices: \(thunderboltDevices.joined(separator: ", "))")

        // 2. Get bridge services from network service order
        guard let bridgeServices = getBridgeServices(), !bridgeServices.isEmpty else {
            logger.debug("No bridge services found")
            return nil
        }
        logger.debug("Found bridge services: \(bridgeServices.keys.joined(separator: ", "))")

        // 3. Find a bridge that contains Thunderbolt interfaces
        for (bridgeDevice, serviceName) in bridgeServices {
            let members = getBridgeMembers(bridgeDevice: bridgeDevice)
            logger.debug("Bridge \(bridgeDevice) (\(serviceName)) has members: \(members.joined(separator: ", "))")

            // Check if any Thunderbolt device is a member of this bridge
            if !members.isDisjoint(with: thunderboltDevices) {
                logger.info("Found Thunderbolt Bridge service: '\(serviceName)' (device: \(bridgeDevice))")
                return serviceName
            }
        }

        logger.debug("No bridge found containing Thunderbolt interfaces")
        return nil
    }

    /// Get Thunderbolt interface device names (e.g., en2, en3) from hardware ports.
    private static func getThunderboltDevices() -> Set<String>? {
        let result = runCommand(["networksetup", "-listallhardwareports"])
        guard result.exitCode == 0 else {
            logger.warning("networksetup -listallhardwareports failed: \(result.error)")
            return nil
        }

        var thunderboltDevices: Set<String> = []
        var currentPort: String?

        for line in result.output.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("Hardware Port:") {
                currentPort = String(trimmed.dropFirst("Hardware Port:".count)).trimmingCharacters(
                    in: .whitespaces)
            } else if trimmed.hasPrefix("Device:"), let port = currentPort {
                let device = String(trimmed.dropFirst("Device:".count)).trimmingCharacters(
                    in: .whitespaces)
                if port.lowercased().contains("thunderbolt") {
                    thunderboltDevices.insert(device)
                }
                currentPort = nil
            }
        }

        return thunderboltDevices
    }

    /// Get mapping of bridge device -> service name from network service order.
    private static func getBridgeServices() -> [String: String]? {
        let result = runCommand(["networksetup", "-listnetworkserviceorder"])
        guard result.exitCode == 0 else {
            logger.warning("networksetup -listnetworkserviceorder failed: \(result.error)")
            return nil
        }

        // Parse service order to find bridge devices and their service names
        // Format: "(1) Service Name\n(Hardware Port: ..., Device: bridge0)\n"
        var bridgeServices: [String: String] = [:]
        var currentService: String?

        for line in result.output.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Match "(N) Service Name" or "(*) Service Name" (disabled)
            // but NOT "(Hardware Port: ...)" lines
            if trimmed.hasPrefix("("), trimmed.contains(")"),
                !trimmed.hasPrefix("(Hardware Port:")
            {
                if let parenEnd = trimmed.firstIndex(of: ")") {
                    let afterParen = trimmed.index(after: parenEnd)
                    if afterParen < trimmed.endIndex {
                        currentService =
                            String(trimmed[afterParen...])
                            .trimmingCharacters(in: .whitespaces)
                    }
                }
            }
            // Match "(Hardware Port: ..., Device: bridgeX)"
            else if let service = currentService, trimmed.contains("Device: bridge") {
                // Extract device name from "..., Device: bridge0)"
                if let deviceRange = trimmed.range(of: "Device: ") {
                    let afterDevice = trimmed[deviceRange.upperBound...]
                    if let parenIndex = afterDevice.firstIndex(of: ")") {
                        let device = String(afterDevice[..<parenIndex])
                        bridgeServices[device] = service
                    }
                }
            }
        }

        return bridgeServices
    }

    /// Get member interfaces of a bridge device via ifconfig.
    private static func getBridgeMembers(bridgeDevice: String) -> Set<String> {
        let result = runCommand(["ifconfig", bridgeDevice])
        guard result.exitCode == 0 else {
            logger.debug("ifconfig \(bridgeDevice) failed")
            return []
        }

        var members: Set<String> = []
        for line in result.output.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("member:") {
                let parts = trimmed.split(separator: " ")
                if parts.count > 1 {
                    members.insert(String(parts[1]))
                }
            }
        }

        return members
    }

    /// Check if a network service is enabled.
    static func isServiceEnabled(serviceName: String) -> Bool? {
        let result = runCommand(["networksetup", "-getnetworkserviceenabled", serviceName])
        guard result.exitCode == 0 else {
            logger.warning("Failed to check if '\(serviceName)' is enabled: \(result.error)")
            return nil
        }

        let output = result.output.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if output.contains("enabled") {
            return true
        }
        if output.contains("disabled") {
            return false
        }
        return nil
    }

    private static func runCommand(_ arguments: [String]) -> CommandResult {
        let process = Process()
        process.launchPath = "/usr/bin/env"
        process.arguments = arguments

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        do {
            try process.run()
        } catch {
            return CommandResult(exitCode: -1, output: "", error: error.localizedDescription)
        }
        process.waitUntilExit()

        let outputData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errorData = stderr.fileHandleForReading.readDataToEndOfFile()

        return CommandResult(
            exitCode: process.terminationStatus,
            output: String(decoding: outputData, as: UTF8.self),
            error: String(decoding: errorData, as: UTF8.self)
        )
    }
}
