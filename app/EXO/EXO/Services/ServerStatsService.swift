import Combine
import Foundation

/// Aggregated server stats from `/v1/stats`. Mirrors the Pydantic
/// ServerStatsResponse shape on the API (camelCase via FrozenModel alias).
struct ServerStats: Decodable, Equatable {
    let uptimeSeconds: Double
    let totalRequests: Int
    let instanceCount: Int
    let nodeCount: Int
    let activeCommands: Int
}

/// Polls `/v1/stats` for the menubar's live numbers (requests served, uptime).
@MainActor
final class ServerStatsService: ObservableObject {
    @Published private(set) var latest: ServerStats?
    @Published private(set) var lastError: String?

    private var timer: Timer?
    private let endpoint: URL
    private let session: URLSession
    private let decoder: JSONDecoder

    init(
        baseURL: URL = URL(string: "http://127.0.0.1:52415")!,
        session: URLSession = .shared
    ) {
        self.endpoint = baseURL.appendingPathComponent("v1/stats")
        self.session = session
        let decoder = JSONDecoder()
        // FrozenModel uses to_camel — keys are already camelCase, so use the
        // default strategy. (Don't apply convertFromSnakeCase here.)
        self.decoder = decoder
    }

    func startPolling(interval: TimeInterval = 2.0) {
        stopPolling()
        Task { await fetchOnce() }
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { await self?.fetchOnce() }
        }
    }

    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }

    func reset() {
        latest = nil
        lastError = nil
    }

    private func fetchOnce() async {
        do {
            var request = URLRequest(url: endpoint)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse,
                (200..<300).contains(http.statusCode)
            else {
                return
            }
            let stats = try decoder.decode(ServerStats.self, from: data)
            latest = stats
            lastError = nil
        } catch {
            // Keep previous value; only surface error on persistent failure
            // if a UI surface ever wants it.
            lastError = error.localizedDescription
        }
    }
}
