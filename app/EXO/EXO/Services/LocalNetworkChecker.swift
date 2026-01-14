import Foundation
import Network
import os.log

/// Checks if the app's local network permission is actually functional.
///
/// macOS local network permission can appear enabled in System Preferences but not
/// actually work after a restart. This service uses NWConnection to mDNS multicast
/// to verify actual connectivity.
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
    private static let hasCompletedInitialCheckKey = "LocalNetworkChecker.hasCompletedInitialCheck"

    @Published private(set) var status: Status = .unknown

    private var connection: NWConnection?
    private var checkTask: Task<Void, Never>?

    /// Whether we've completed at least one check (stored in UserDefaults)
    private var hasCompletedInitialCheck: Bool {
        get { UserDefaults.standard.bool(forKey: Self.hasCompletedInitialCheckKey) }
        set { UserDefaults.standard.set(newValue, forKey: Self.hasCompletedInitialCheckKey) }
    }

    /// Checks if local network access is working.
    func check() {
        checkTask?.cancel()
        status = .checking

        // Use longer timeout on first launch to allow time for permission prompt
        let isFirstCheck = !hasCompletedInitialCheck
        let timeout: UInt64 = isFirstCheck ? 30_000_000_000 : 3_000_000_000

        checkTask = Task { [weak self] in
            guard let self else { return }

            Self.logger.info("Checking local network connectivity (first check: \(isFirstCheck))")
            let result = await self.checkConnectivity(timeout: timeout)
            self.status = result
            self.hasCompletedInitialCheck = true

            Self.logger.info("Local network check complete: \(result.displayText)")
        }
    }

    /// Checks connectivity using NWConnection to mDNS multicast.
    /// The connection attempt triggers the permission prompt if not yet shown.
    private func checkConnectivity(timeout: UInt64) async -> Status {
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

            conn.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    resumeOnce(.working)
                case .waiting(let error):
                    let errorStr = "\(error)"
                    if errorStr.contains("54") || errorStr.contains("ECONNRESET") {
                        resumeOnce(.notWorking(reason: "Connection blocked"))
                    }
                // Otherwise keep waiting - might be showing permission prompt
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
                try? await Task.sleep(nanoseconds: timeout)
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
