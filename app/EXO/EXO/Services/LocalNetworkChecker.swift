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

        /// macOS said Local Network access is off for EXO.
        static let deniedByMacOSReason = "Local Network access is off"
        /// The probe failed in a way that usually, but not always, means
        /// access is off: "No route to host" also happens for a moment while
        /// the network comes up.
        static let deniedReason = "Permission denied"
        static let blockedReason = "Connection blocked"

        var suggestsAccessOff: Bool {
            guard case .notWorking(let reason) = self else { return false }
            return [Self.deniedByMacOSReason, Self.deniedReason, Self.blockedReason]
                .contains(reason)
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

    /// Whether to tell the user Local Network access is off: only when two
    /// checks in a row say so. Right after exo starts, macOS can report
    /// access as off for a few seconds before the check works (seen on a Mac
    /// Studio with access allowed), and a network still coming up at login
    /// looks the same; one check shouldn't flash a warning.
    nonisolated static func accessIsOff(current: Status, previous: Status) -> Bool {
        current.suggestsAccessOff && previous.suggestsAccessOff
    }

    private static let logger = Logger(subsystem: "io.exo.EXO", category: "LocalNetworkChecker")
    private static let hasCompletedInitialCheckKey = "LocalNetworkChecker.hasCompletedInitialCheck"

    @Published private(set) var status: Status = .unknown
    /// What the menu bar shows; see `accessIsOff(current:previous:)`.
    @Published private(set) var accessIsOff = false

    private var connection: NWConnection?
    private var checkTask: Task<Void, Never>?
    private var periodicTask: Task<Void, Never>?

    /// Whether we've completed at least one check (stored in UserDefaults)
    private var hasCompletedInitialCheck: Bool {
        get { UserDefaults.standard.bool(forKey: Self.hasCompletedInitialCheckKey) }
        set { UserDefaults.standard.set(newValue, forKey: Self.hasCompletedInitialCheckKey) }
    }

    /// Checks if local network access is working (one-time check).
    func check() {
        performCheck()
    }

    /// Starts periodic checking of local network access.
    /// Re-checks every `interval` seconds so the warning disappears when user grants permission.
    func startPeriodicChecking(interval: TimeInterval = 10) {
        stopPeriodicChecking()
        // Do an immediate check first
        performCheck()
        // Then schedule periodic checks
        periodicTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                guard !Task.isCancelled else { break }
                self?.performCheck()
            }
        }
    }

    /// Stops periodic checking.
    func stopPeriodicChecking() {
        periodicTask?.cancel()
        periodicTask = nil
    }

    private func performCheck() {
        checkTask?.cancel()
        // Only show "checking" status on first check to avoid UI flicker
        if status == .unknown {
            status = .checking
        }

        // Use longer timeout on first launch to allow time for permission prompt
        let isFirstCheck = !hasCompletedInitialCheck
        let timeout: UInt64 = isFirstCheck ? 30_000_000_000 : 3_000_000_000

        checkTask = Task { [weak self] in
            guard let self else { return }

            Self.logger.debug("Checking local network connectivity (first check: \(isFirstCheck))")
            let result = await self.checkConnectivity(timeout: timeout)
            // A newer check replaced this one; its result is out of date.
            guard !Task.isCancelled else { return }
            let previous = self.status
            self.status = result
            self.accessIsOff = Self.accessIsOff(current: result, previous: previous)
            self.hasCompletedInitialCheck = true

            // Only log on state changes or first check to reduce noise
            if isFirstCheck || result != previous {
                Self.logger.notice(
                    "Local network check: \(result.displayText, privacy: .public)")
            }
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

            conn.stateUpdateHandler = { [weak conn] state in
                switch state {
                case .ready:
                    resumeOnce(.working)
                case .waiting(let error):
                    let errorStr = "\(error)"
                    if conn?.currentPath?.unsatisfiedReason == .localNetworkDenied {
                        // What macOS reports when Local Network access is off
                        // for EXO (TN3179).
                        resumeOnce(.notWorking(reason: Status.deniedByMacOSReason))
                    } else if errorStr.contains("54") || errorStr.contains("ECONNRESET") {
                        resumeOnce(.notWorking(reason: Status.blockedReason))
                    }
                // Otherwise keep waiting - might be showing permission prompt
                case .failed(let error):
                    let errorStr = "\(error)"
                    if errorStr.contains("65") || errorStr.contains("EHOSTUNREACH")
                        || errorStr.contains("permission") || errorStr.contains("denied")
                    {
                        resumeOnce(.notWorking(reason: Status.deniedReason))
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
        stopPeriodicChecking()
        checkTask?.cancel()
        checkTask = nil
        connection?.cancel()
        connection = nil
    }
}
