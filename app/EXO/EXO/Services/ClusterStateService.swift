import Combine
import Foundation

@MainActor
final class ClusterStateService: ObservableObject {
    @Published private(set) var latestSnapshot: ClusterState?
    @Published private(set) var lastError: String?
    @Published private(set) var lastActionMessage: String?
    @Published private(set) var modelOptions: [ModelOption] = []
    @Published private(set) var localNodeId: String?
    /// Whether exo is answering. Once it stops, `latestSnapshot` is cleared
    /// rather than left showing the last stats it reported.
    @Published private(set) var apiState: APIResponsiveness.State = .connecting

    /// A poll that gets no answer in this time counts as failed, so a hung exo
    /// is noticed in seconds rather than after URLSession's 60-second default.
    static let pollTimeout: TimeInterval = 10

    private var timer: Timer?
    private var responsiveness = APIResponsiveness()
    private var launchObservation: AnyCancellable?
    private let decoder: JSONDecoder
    private let session: URLSession
    private let baseURL: URL
    private let endpoint: URL

    init(
        baseURL: URL = URL(string: "http://127.0.0.1:52415")!,
        session: URLSession = ClusterStateService.makeNonCachingSession()
    ) {
        self.baseURL = baseURL
        self.endpoint = baseURL.appendingPathComponent("state")
        self.session = session
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.decoder = decoder
    }

    /// `URLSession.shared` carries an on-disk `URLCache` that persists every
    /// response body under `~/Library/Caches/exolabs.EXO/`. We poll `/state`
    /// at 2 Hz from `startPolling`, so leaving the shared cache attached
    /// dirties ~500–620 KB/sec of file-backed memory and trips macOS's
    /// per-process `disk writes` resource limit (microstackshot reports
    /// observed on M3 Ultra producing GBs of cached responses per hour).
    /// Cluster-state polling responses are time-sensitive and small; they
    /// gain nothing from being cached on disk. Use an ephemeral session
    /// with `urlCache = nil` so neither response bodies nor metadata
    /// touch disk.
    nonisolated private static func makeNonCachingSession() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.urlCache = nil
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config)
    }

    func startPolling(interval: TimeInterval = 0.5) {
        stopPolling()
        Task {
            await fetchLocalNodeId()
            await fetchModels()
            await fetchSnapshot()
        }
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { await self?.fetchSnapshot() }
        }
    }

    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }

    func resetTransientState() {
        latestSnapshot = nil
        lastError = nil
        lastActionMessage = nil
        localNodeId = nil
        responsiveness = APIResponsiveness()
        apiState = responsiveness.state
    }

    /// Forgets what the previous run reported each time exo is started, so a
    /// restart (e.g. Save & Restart in Settings) begins at "connecting"
    /// instead of keeping old stats or calling the new exo unresponsive before
    /// it has had a chance to answer.
    func resetWhenExoStarts<StatusPublisher: Publisher>(_ statuses: StatusPublisher)
    where
        StatusPublisher.Output == ExoProcessController.Status,
        StatusPublisher.Failure == Never
    {
        launchObservation =
            statuses
            .removeDuplicates()
            .filter { $0 == .starting }
            .sink { [weak self] _ in
                self?.resetTransientState()
            }
    }

    private func fetchLocalNodeId() async {
        do {
            let url = baseURL.appendingPathComponent("node_id")
            var request = URLRequest(url: url)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            request.timeoutInterval = Self.pollTimeout
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                (200..<300).contains(httpResponse.statusCode)
            else {
                return
            }
            if let nodeId = try? decoder.decode(String.self, from: data) {
                localNodeId = nodeId
            }
        } catch {
            // Silently ignore - localNodeId will remain nil and retry on next poll
        }
    }

    func fetchSnapshot() async {
        // Retry fetching local node ID if not yet set
        if localNodeId == nil {
            await fetchLocalNodeId()
        }
        let data: Data
        do {
            var request = URLRequest(url: endpoint)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            request.timeoutInterval = Self.pollTimeout
            let (body, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw URLError(.badServerResponse)
            }
            guard (200..<300).contains(httpResponse.statusCode) else {
                throw URLError(.badServerResponse)
            }
            data = body
        } catch {
            lastError = error.localizedDescription
            recordPoll(answered: false)
            return
        }
        recordPoll(answered: true)
        do {
            let snapshot = try decoder.decode(ClusterState.self, from: data)
            latestSnapshot = snapshot
            if modelOptions.isEmpty {
                Task { await fetchModels() }
            }
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }

    private func recordPoll(answered: Bool) {
        if answered {
            responsiveness.recordAnswer()
        } else {
            responsiveness.recordFailure()
        }
        if responsiveness.state == .notResponding, latestSnapshot != nil {
            latestSnapshot = nil
        }
        if apiState != responsiveness.state {
            apiState = responsiveness.state
        }
    }

    func deleteInstance(_ id: String) async {
        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("instance/\(id)"))
            request.httpMethod = "DELETE"
            request.setValue("application/json", forHTTPHeaderField: "Accept")
            let (_, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw URLError(.badServerResponse)
            }
            guard (200..<300).contains(httpResponse.statusCode) else {
                throw URLError(.badServerResponse)
            }
            lastActionMessage = "Instance deleted"
            await fetchSnapshot()
        } catch {
            lastError = "Failed to delete instance: \(error.localizedDescription)"
        }
    }

    func launchInstance(modelId: String, sharding: String, instanceMeta: String, minNodes: Int)
        async
    {
        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("instance"))
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let payload: [String: Any] = [
                "model_id": modelId,
                "sharding": sharding,
                "instance_meta": instanceMeta,
                "min_nodes": minNodes,
            ]
            request.httpBody = try JSONSerialization.data(withJSONObject: payload, options: [])
            let (_, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw URLError(.badServerResponse)
            }
            guard (200..<300).contains(httpResponse.statusCode) else {
                throw URLError(.badServerResponse)
            }
            lastActionMessage = "Instance launched"
            await fetchSnapshot()
        } catch {
            lastError = "Failed to launch instance: \(error.localizedDescription)"
        }
    }

    func fetchModels() async {
        do {
            let url = baseURL.appendingPathComponent("models")
            let (data, response) = try await session.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse,
                (200..<300).contains(httpResponse.statusCode)
            else {
                throw URLError(.badServerResponse)
            }
            let list = try decoder.decode(ModelListResponse.self, from: data)
            modelOptions = list.data.map { ModelOption(id: $0.id, displayName: $0.name ?? $0.id) }
        } catch {
            lastError = "Failed to load models: \(error.localizedDescription)"
        }
    }
}

struct ModelOption: Identifiable {
    let id: String
    let displayName: String
}

struct ModelListResponse: Decodable {
    let data: [ModelListModel]
}

struct ModelListModel: Decodable {
    let id: String
    let name: String?
}
