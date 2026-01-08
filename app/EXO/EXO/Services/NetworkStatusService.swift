import AppKit
import Foundation

@MainActor
final class NetworkStatusService: ObservableObject {
    @Published private(set) var status: NetworkStatus = .empty
    private var timer: Timer?

    func refresh() async {
        let fetched = await Task.detached(priority: .background) {
            NetworkStatusFetcher().fetch()
        }.value
        status = fetched
    }

    func startPolling(interval: TimeInterval = 30) {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { await self.refresh() }
        }
        if let timer {
            RunLoop.main.add(timer, forMode: .common)
        }
        Task { await refresh() }
    }

    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }
}

struct NetworkStatus: Equatable {
    let thunderboltBridgeState: ThunderboltState?
    let bridgeInactive: Bool?
    let interfaceStatuses: [InterfaceIpStatus]
    let rdmaStatus: RDMAStatus

    static let empty = NetworkStatus(
        thunderboltBridgeState: nil,
        bridgeInactive: nil,
        interfaceStatuses: [],
        rdmaStatus: .empty
    )
}

struct RDMAStatus: Equatable {
    let rdmaCtlEnabled: Bool?
    let devices: [String]
    let activePorts: [RDMAPort]

    var isAvailable: Bool {
        rdmaCtlEnabled == true || !devices.isEmpty
    }

    static let empty = RDMAStatus(rdmaCtlEnabled: nil, devices: [], activePorts: [])
}

struct RDMAPort: Equatable {
    let device: String
    let port: String
    let state: String
}

struct InterfaceIpStatus: Equatable {
    let interfaceName: String
    let ipAddress: String?
}

enum ThunderboltState: Equatable {
    case enabled
    case disabled
    case deleted
}

private struct NetworkStatusFetcher {
    func fetch() -> NetworkStatus {
        NetworkStatus(
            thunderboltBridgeState: readThunderboltBridgeState(),
            bridgeInactive: readBridgeInactive(),
            interfaceStatuses: readInterfaceStatuses(),
            rdmaStatus: readRDMAStatus()
        )
    }

    private func readRDMAStatus() -> RDMAStatus {
        let rdmaCtlEnabled = readRDMACtlEnabled()
        let devices = readRDMADevices()
        let activePorts = readRDMAActivePorts()
        return RDMAStatus(
            rdmaCtlEnabled: rdmaCtlEnabled, devices: devices, activePorts: activePorts)
    }

    private func readRDMACtlEnabled() -> Bool? {
        let result = runCommand(["rdma_ctl", "status"])
        guard result.exitCode == 0 else { return nil }
        let output = result.output.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if output.contains("enabled") {
            return true
        }
        if output.contains("disabled") {
            return false
        }
        return nil
    }

    private func readRDMADevices() -> [String] {
        let result = runCommand(["ibv_devices"])
        guard result.exitCode == 0 else { return [] }
        var devices: [String] = []
        for line in result.output.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("---") || trimmed.lowercased().hasPrefix("device")
                || trimmed.isEmpty
            {
                continue
            }
            let parts = trimmed.split(separator: " ", maxSplits: 1)
            if let deviceName = parts.first {
                devices.append(String(deviceName))
            }
        }
        return devices
    }

    private func readRDMAActivePorts() -> [RDMAPort] {
        let result = runCommand(["ibv_devinfo"])
        guard result.exitCode == 0 else { return [] }
        var ports: [RDMAPort] = []
        var currentDevice: String?
        var currentPort: String?

        for line in result.output.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("hca_id:") {
                currentDevice = trimmed.replacingOccurrences(of: "hca_id:", with: "")
                    .trimmingCharacters(in: .whitespaces)
            } else if trimmed.hasPrefix("port:") {
                currentPort = trimmed.replacingOccurrences(of: "port:", with: "")
                    .trimmingCharacters(in: .whitespaces)
            } else if trimmed.hasPrefix("state:") {
                let state = trimmed.replacingOccurrences(of: "state:", with: "").trimmingCharacters(
                    in: .whitespaces)
                if let device = currentDevice, let port = currentPort {
                    if state.lowercased().contains("active") {
                        ports.append(RDMAPort(device: device, port: port, state: state))
                    }
                }
            }
        }
        return ports
    }

    private func readThunderboltBridgeState() -> ThunderboltState? {
        let result = runCommand(["networksetup", "-getnetworkserviceenabled", "Thunderbolt Bridge"])
        guard result.exitCode == 0 else {
            let lower = result.output.lowercased() + result.error.lowercased()
            if lower.contains("not a recognized network service") {
                return .deleted
            }
            return nil
        }
        let output = result.output.lowercased()
        if output.contains("enabled") {
            return .enabled
        }
        if output.contains("disabled") {
            return .disabled
        }
        return nil
    }

    private func readBridgeInactive() -> Bool? {
        let result = runCommand(["ifconfig", "bridge0"])
        guard result.exitCode == 0 else { return nil }
        guard
            let statusLine = result.output
                .components(separatedBy: .newlines)
                .first(where: { $0.contains("status:") })?
                .lowercased()
        else {
            return nil
        }
        if statusLine.contains("inactive") {
            return true
        }
        if statusLine.contains("active") {
            return false
        }
        return nil
    }

    private func readInterfaceStatuses() -> [InterfaceIpStatus] {
        (0...7).map { "en\($0)" }.map(readInterfaceStatus)
    }

    private func readInterfaceStatus(for interface: String) -> InterfaceIpStatus {
        let result = runCommand(["ifconfig", interface])
        guard result.exitCode == 0 else {
            return InterfaceIpStatus(
                interfaceName: interface,
                ipAddress: nil
            )
        }

        let output = result.output
        let ip = firstInet(from: output)

        return InterfaceIpStatus(
            interfaceName: interface,
            ipAddress: ip
        )
    }

    private func firstInet(from ifconfigOutput: String) -> String? {
        for line in ifconfigOutput.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("inet ") else { continue }
            let parts = trimmed.split(separator: " ")
            if parts.count >= 2 {
                let candidate = String(parts[1])
                if candidate != "127.0.0.1" {
                    return candidate
                }
            }
        }
        return nil
    }

    private struct CommandResult {
        let exitCode: Int32
        let output: String
        let error: String
    }

    private func runCommand(_ arguments: [String]) -> CommandResult {
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
