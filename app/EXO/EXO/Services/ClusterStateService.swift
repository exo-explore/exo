import Combine
import Foundation

@MainActor
final class ClusterStateService: ObservableObject {
    @Published private(set) var latestSnapshot: ClusterState?
    @Published private(set) var lastError: String?
    @Published private(set) var lastActionMessage: String?
    @Published private(set) var modelOptions: [ModelOption] = []
    @Published private(set) var localNodeId: String?

    private var timer: Timer?
    private let decoder: JSONDecoder
    private let session: URLSession
    private let baseURL: URL
    private let endpoint: URL

    init(
        baseURL: URL = URL(string: "http://127.0.0.1:52415")!,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.endpoint = baseURL.appendingPathComponent("state")
        self.session = session
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.decoder = decoder
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
    }

    private func fetchLocalNodeId() async {
        do {
            let url = baseURL.appendingPathComponent("node_id")
            var request = URLRequest(url: url)
            request.cachePolicy = .reloadIgnoringLocalCacheData
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

    private func fetchSnapshot() async {
        // Retry fetching local node ID if not yet set
        if localNodeId == nil {
            await fetchLocalNodeId()
        }
        do {
            var request = URLRequest(url: endpoint)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw URLError(.badServerResponse)
            }
            guard (200..<300).contains(httpResponse.statusCode) else {
                throw URLError(.badServerResponse)
            }
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
