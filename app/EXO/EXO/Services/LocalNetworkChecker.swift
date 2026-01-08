import Foundation
import Network
import os.log

/// Checks if the app's local network permission is actually functional.
///
/// macOS local network permission can appear enabled in System Preferences but not
/// actually work after a restart. This service detects this by creating a UDP
/// connection to the mDNS multicast address (224.0.0.251:5353).
@MainActor
final class LocalNetworkChecker: ObservableObject {
    enum Status: Equatable {
        case unknown
        case checking
        case working
        case notWorking(reason: String)

        var isHealthy: Bool {
            if case .working = self { return true }
            return false
        }

        var displayText: String {
            switch self {
            case .unknown:
                return "Unknown"
            case .checking:
                return "Checking..."
            case .working:
                return "Working"
            case .notWorking(let reason):
                return reason
            }
        }
    }

    private static let logger = Logger(subsystem: "io.exo.EXO", category: "LocalNetworkChecker")

    @Published private(set) var status: Status = .unknown
    @Published private(set) var lastConnectionState: String = "none"

    private var connection: NWConnection?
    private var checkTask: Task<Void, Never>?

    /// Checks if local network access is working.
    func check() {
        checkTask?.cancel()
        status = .checking
        lastConnectionState = "connecting"

        checkTask = Task { [weak self] in
            guard let self else { return }
            let result = await self.performCheck()
            self.status = result
            Self.logger.info("Local network check complete: \(result.displayText)")
        }
    }

    private func performCheck() async -> Status {
        Self.logger.info("Checking local network access via UDP multicast")

        connection?.cancel()
        connection = nil

        // mDNS multicast address - same as libp2p uses for peer discovery
        let host = NWEndpoint.Host("224.0.0.251")
        let port = NWEndpoint.Port(integerLiteral: 5353)

        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true

        let conn = NWConnection(host: host, port: port, using: params)
        connection = conn

        return await withCheckedContinuation { continuation in
            var hasResumed = false
            let lock = NSLock()

            let resumeOnce: (Status) -> Void = { status in
                lock.lock()
                defer { lock.unlock() }
                guard !hasResumed else { return }
                hasResumed = true
                continuation.resume(returning: status)
            }

            conn.stateUpdateHandler = { [weak self] state in
                let stateStr: String
                switch state {
                case .setup: stateStr = "setup"
                case .preparing: stateStr = "preparing"
                case .ready: stateStr = "ready"
                case .waiting(let e): stateStr = "waiting(\(e))"
                case .failed(let e): stateStr = "failed(\(e))"
                case .cancelled: stateStr = "cancelled"
                @unknown default: stateStr = "unknown"
                }

                Task { @MainActor in
                    self?.lastConnectionState = stateStr
                }

                switch state {
                case .ready:
                    resumeOnce(.working)
                case .waiting(let error):
                    let errorStr = "\(error)"
                    if errorStr.contains("54") || errorStr.contains("ECONNRESET") {
                        resumeOnce(.notWorking(reason: "Connection blocked"))
                    }
                case .failed(let error):
                    let errorStr = "\(error)"
                    if errorStr.contains("65") || errorStr.contains("EHOSTUNREACH")
                        || errorStr.contains("permission") || errorStr.contains("denied")
                    {
                        resumeOnce(.notWorking(reason: "Permission denied"))
                    } else {
                        resumeOnce(.notWorking(reason: "Failed: \(error.localizedDescription)"))
                    }
                case .cancelled, .setup, .preparing:
                    break
                @unknown default:
                    break
                }
            }

            conn.start(queue: .main)

            Task {
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                let state = conn.state
                switch state {
                case .ready:
                    resumeOnce(.working)
                case .waiting, .preparing, .setup:
                    resumeOnce(.notWorking(reason: "Timeout (may be blocked)"))
                default:
                    resumeOnce(.notWorking(reason: "Timeout"))
                }
            }
        }
    }

    func stop() {
        checkTask?.cancel()
        checkTask = nil
        connection?.cancel()
        connection = nil
    }
}
