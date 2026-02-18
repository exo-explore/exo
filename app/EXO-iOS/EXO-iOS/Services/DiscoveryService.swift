import Foundation
import Network
import os

struct DiscoveredCluster: Identifiable, Equatable {
    let id: String
    let name: String
    let endpoint: NWEndpoint

    static func == (lhs: DiscoveredCluster, rhs: DiscoveredCluster) -> Bool {
        lhs.id == rhs.id && lhs.name == rhs.name
    }
}

@Observable
@MainActor
final class DiscoveryService {
    private(set) var discoveredClusters: [DiscoveredCluster] = []
    private(set) var isSearching = false

    private var browser: NWBrowser?

    func startBrowsing() {
        guard browser == nil else { return }

        let browser = NWBrowser(for: .bonjour(type: "_exo._tcp", domain: nil), using: .tcp)

        browser.stateUpdateHandler = { [weak self] state in
            guard let service = self else { return }
            Task { @MainActor in
                switch state {
                case .ready:
                    service.isSearching = true
                case .failed, .cancelled:
                    service.isSearching = false
                default:
                    break
                }
            }
        }

        browser.browseResultsChangedHandler = { [weak self] results, _ in
            guard let service = self else { return }
            Task { @MainActor in
                service.discoveredClusters = results.compactMap { result in
                    guard case .service(let name, _, _, _) = result.endpoint else {
                        return nil
                    }
                    return DiscoveredCluster(
                        id: name,
                        name: name,
                        endpoint: result.endpoint
                    )
                }
            }
        }

        browser.start(queue: .main)
        self.browser = browser
    }

    func stopBrowsing() {
        browser?.cancel()
        browser = nil
        isSearching = false
        discoveredClusters = []
    }

    /// Resolve a discovered Bonjour endpoint to an IP address and port, then return a ConnectionInfo.
    func resolve(_ cluster: DiscoveredCluster) async -> ConnectionInfo? {
        await withCheckedContinuation { continuation in
            let didResume = OSAllocatedUnfairLock(initialState: false)
            let connection = NWConnection(to: cluster.endpoint, using: .tcp)
            connection.stateUpdateHandler = { state in
                guard
                    didResume.withLock({
                        guard !$0 else { return false }
                        $0 = true
                        return true
                    })
                else { return }
                switch state {
                case .ready:
                    if let innerEndpoint = connection.currentPath?.remoteEndpoint,
                        case .hostPort(let host, let port) = innerEndpoint
                    {
                        var hostString: String
                        switch host {
                        case .ipv4(let addr):
                            hostString = "\(addr)"
                        case .ipv6(let addr):
                            hostString = "\(addr)"
                        case .name(let name, _):
                            hostString = name
                        @unknown default:
                            hostString = "\(host)"
                        }
                        // Strip interface scope suffix (e.g. "%en0")
                        if let pct = hostString.firstIndex(of: "%") {
                            hostString = String(hostString[..<pct])
                        }
                        let info = ConnectionInfo(
                            host: hostString,
                            port: Int(port.rawValue),
                            nodeId: nil
                        )
                        connection.cancel()
                        continuation.resume(returning: info)
                    } else {
                        connection.cancel()
                        continuation.resume(returning: nil)
                    }
                case .failed, .cancelled:
                    continuation.resume(returning: nil)
                default:
                    // Not a terminal state — allow future callbacks
                    didResume.withLock { $0 = false }
                }
            }
            connection.start(queue: .global(qos: .userInitiated))
        }
    }
}
