import Foundation

enum ConnectionState: Equatable {
    case disconnected
    case connecting
    case connected(ConnectionInfo)
}

struct ModelOption: Identifiable, Equatable {
    let id: String
    let displayName: String
}

@Observable
@MainActor
final class ClusterService {
    private(set) var connectionState: ConnectionState = .disconnected
    private(set) var availableModels: [ModelOption] = []
    private(set) var lastError: String?

    private let session: URLSession
    private let decoder: JSONDecoder
    private var pollingTask: Task<Void, Never>?

    private static let connectionInfoKey = "exo_last_connection_info"

    var isConnected: Bool {
        if case .connected = connectionState { return true }
        return false
    }

    var currentConnection: ConnectionInfo? {
        if case .connected(let info) = connectionState { return info }
        return nil
    }

    init(session: URLSession = .shared) {
        self.session = session
        let decoder = JSONDecoder()
        self.decoder = decoder
    }

    // MARK: - Connection

    func connect(to info: ConnectionInfo) async {
        connectionState = .connecting
        lastError = nil

        do {
            let url = info.baseURL.appendingPathComponent("node_id")
            var request = URLRequest(url: url)
            request.timeoutInterval = 5
            request.cachePolicy = .reloadIgnoringLocalCacheData
            let (_, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse,
                (200..<300).contains(httpResponse.statusCode)
            else {
                throw URLError(.badServerResponse)
            }

            connectionState = .connected(info)
            persistConnection(info)
            startPolling()
            await fetchModels(baseURL: info.baseURL)
        } catch {
            connectionState = .disconnected
            lastError = "Could not connect to \(info.host):\(info.port)"
        }
    }

    func connectToDiscoveredCluster(_ cluster: DiscoveredCluster, using discoveryService: DiscoveryService) async {
        guard case .disconnected = connectionState else { return }
        connectionState = .connecting
        lastError = nil

        guard let info = await discoveryService.resolve(cluster) else {
            connectionState = .disconnected
            lastError = "Could not resolve \(cluster.name)"
            return
        }
        connectionState = .disconnected  // reset so connect() can proceed
        await connect(to: info)
    }

    func disconnect() {
        stopPolling()
        connectionState = .disconnected
        availableModels = []
        lastError = nil
    }

    func attemptAutoReconnect() async {
        guard case .disconnected = connectionState,
            let info = loadPersistedConnection()
        else { return }
        await connect(to: info)
    }

    // MARK: - Polling

    private func startPolling(interval: TimeInterval = 2.0) {
        stopPolling()
        pollingTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(interval))
                guard let self, !Task.isCancelled else { return }
                guard let connection = self.currentConnection else { return }
                await self.fetchModels(baseURL: connection.baseURL)
            }
        }
    }

    private func stopPolling() {
        pollingTask?.cancel()
        pollingTask = nil
    }

    // MARK: - API

    private func fetchModels(baseURL: URL) async {
        do {
            let url = baseURL.appendingPathComponent("models")
            var request = URLRequest(url: url)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse,
                (200..<300).contains(httpResponse.statusCode)
            else { return }

            let list = try decoder.decode(ModelListResponse.self, from: data)
            availableModels = list.data.map {
                ModelOption(id: $0.id, displayName: $0.name ?? $0.id)
            }
        } catch {
            // Models fetch failed silently — will retry on next poll
        }
    }

    func streamChatCompletion(request body: ChatCompletionRequest) -> AsyncThrowingStream<
        ChatCompletionChunk, Error
    > {
        AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self, let connection = self.currentConnection else {
                    continuation.finish(throwing: URLError(.notConnectedToInternet))
                    return
                }

                do {
                    let url = connection.baseURL.appendingPathComponent("v1/chat/completions")
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.httpBody = try JSONEncoder().encode(body)

                    let (bytes, response) = try await self.session.bytes(for: request)

                    guard let httpResponse = response as? HTTPURLResponse,
                        (200..<300).contains(httpResponse.statusCode)
                    else {
                        continuation.finish(throwing: URLError(.badServerResponse))
                        return
                    }

                    let parser = SSEStreamParser<ChatCompletionChunk>(
                        bytes: bytes, decoder: self.decoder)
                    for try await chunk in parser {
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - Persistence

    private func persistConnection(_ info: ConnectionInfo) {
        if let data = try? JSONEncoder().encode(info) {
            UserDefaults.standard.set(data, forKey: Self.connectionInfoKey)
        }
    }

    private func loadPersistedConnection() -> ConnectionInfo? {
        guard let data = UserDefaults.standard.data(forKey: Self.connectionInfoKey) else {
            return nil
        }
        return try? JSONDecoder().decode(ConnectionInfo.self, from: data)
    }
}
