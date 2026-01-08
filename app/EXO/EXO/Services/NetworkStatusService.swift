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

    static let empty = NetworkStatus(
        thunderboltBridgeState: nil,
        bridgeInactive: nil,
        interfaceStatuses: []
    )
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
            interfaceStatuses: readInterfaceStatuses()
        )
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
