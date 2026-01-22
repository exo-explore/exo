import Foundation

// MARK: - API payloads

struct ClusterState: Decodable {
    let instances: [String: ClusterInstance]
    let runners: [String: RunnerStatusSummary]
    let tasks: [String: ClusterTask]
    let topology: Topology?
    let downloads: [String: [NodeDownloadStatus]]
    let thunderboltBridgeCycles: [[String]]

    // Granular node state (split from the old nodeProfiles)
    let nodeIdentities: [String: NodeIdentity]
    let nodeMemory: [String: MemoryInfo]
    let nodeSystem: [String: SystemInfo]
    let nodeThunderboltBridge: [String: ThunderboltBridgeStatus]

    /// Computed property for backwards compatibility - merges granular state into NodeProfile
    var nodeProfiles: [String: NodeProfile] {
        var profiles: [String: NodeProfile] = [:]
        let allNodeIds = Set(nodeIdentities.keys)
            .union(nodeMemory.keys)
            .union(nodeSystem.keys)
        for nodeId in allNodeIds {
            let identity = nodeIdentities[nodeId]
            let memory = nodeMemory[nodeId]
            let system = nodeSystem[nodeId]
            profiles[nodeId] = NodeProfile(
                modelId: identity?.modelId,
                chipId: identity?.chipId,
                friendlyName: identity?.friendlyName,
                memory: memory,
                system: system
            )
        }
        return profiles
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rawInstances = try container.decode([String: TaggedInstance].self, forKey: .instances)
        self.instances = rawInstances.mapValues(\.instance)
        self.runners = try container.decode([String: RunnerStatusSummary].self, forKey: .runners)
        let rawTasks =
            try container.decodeIfPresent([String: TaggedTask].self, forKey: .tasks) ?? [:]
        self.tasks = rawTasks.compactMapValues(\.task)
        self.topology = try container.decodeIfPresent(Topology.self, forKey: .topology)
        let rawDownloads =
            try container.decodeIfPresent([String: [TaggedNodeDownload]].self, forKey: .downloads)
            ?? [:]
        self.downloads = rawDownloads.mapValues { $0.compactMap(\.status) }
        self.thunderboltBridgeCycles =
            try container.decodeIfPresent([[String]].self, forKey: .thunderboltBridgeCycles) ?? []

        // Granular node state
        self.nodeIdentities =
            try container.decodeIfPresent([String: NodeIdentity].self, forKey: .nodeIdentities)
            ?? [:]
        self.nodeMemory =
            try container.decodeIfPresent([String: MemoryInfo].self, forKey: .nodeMemory) ?? [:]
        self.nodeSystem =
            try container.decodeIfPresent([String: SystemInfo].self, forKey: .nodeSystem) ?? [:]
        self.nodeThunderboltBridge =
            try container.decodeIfPresent(
                [String: ThunderboltBridgeStatus].self, forKey: .nodeThunderboltBridge
            ) ?? [:]
    }

    private enum CodingKeys: String, CodingKey {
        case instances
        case runners
        case topology
        case tasks
        case downloads
        case thunderboltBridgeCycles
        case nodeIdentities
        case nodeMemory
        case nodeSystem
        case nodeThunderboltBridge
    }
}

private struct TaggedInstance: Decodable {
    let instance: ClusterInstance

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let payloads = try container.decode([String: ClusterInstancePayload].self)
        guard let entry = payloads.first else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath, debugDescription: "Empty instance payload")
            )
        }
        self.instance = ClusterInstance(
            instanceId: entry.value.instanceId,
            shardAssignments: entry.value.shardAssignments,
            sharding: entry.key
        )
    }
}

private struct ClusterInstancePayload: Decodable {
    let instanceId: String?
    let shardAssignments: ShardAssignments
}

struct ClusterInstance {
    let instanceId: String?
    let shardAssignments: ShardAssignments
    let sharding: String
}

struct ShardAssignments: Decodable {
    let modelId: String
    let nodeToRunner: [String: String]
}

struct RunnerStatusSummary: Decodable {
    let status: String
    let errorMessage: String?

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let payloads = try container.decode([String: RunnerStatusDetail].self)
        guard let entry = payloads.first else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath, debugDescription: "Empty runner status payload")
            )
        }
        self.status = entry.key
        self.errorMessage = entry.value.errorMessage
    }
}

struct RunnerStatusDetail: Decodable {
    let errorMessage: String?
}

struct NodeProfile: Decodable {
    let modelId: String?
    let chipId: String?
    let friendlyName: String?
    let memory: MemoryInfo?
    let system: SystemInfo?
}

struct NodeIdentity: Decodable {
    let modelId: String?
    let chipId: String?
    let friendlyName: String?
}

struct ThunderboltBridgeStatus: Decodable {
    let enabled: Bool
    let exists: Bool
    let serviceName: String?
}

struct MemoryInfo: Decodable {
    let ramTotal: MemoryValue?
    let ramAvailable: MemoryValue?
}

struct MemoryValue: Decodable {
    let inBytes: Int64?
}

struct SystemInfo: Decodable {
    let gpuUsage: Double?
    let temp: Double?
    let sysPower: Double?
    let pcpuUsage: Double?
    let ecpuUsage: Double?
}

struct Topology: Decodable {
    /// Node IDs in the topology
    let nodes: [String]
    /// Flattened list of connections (source -> sink pairs)
    let connections: [TopologyConnection]

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.nodes = try container.decodeIfPresent([String].self, forKey: .nodes) ?? []

        // Connections come as nested map: { source: { sink: [edges] } }
        // We flatten to array of (source, sink) pairs
        var flatConnections: [TopologyConnection] = []
        if let nested = try container.decodeIfPresent(
            [String: [String: [AnyCodable]]].self, forKey: .connections
        ) {
            for (source, sinks) in nested {
                for sink in sinks.keys {
                    flatConnections.append(
                        TopologyConnection(localNodeId: source, sendBackNodeId: sink))
                }
            }
        }
        self.connections = flatConnections
    }

    private enum CodingKeys: String, CodingKey {
        case nodes
        case connections
    }
}

/// Placeholder for decoding arbitrary JSON values we don't need to inspect
private struct AnyCodable: Decodable {
    init(from decoder: Decoder) throws {
        // Just consume the value without storing it
        _ = try? decoder.singleValueContainer().decode(Bool.self)
        _ = try? decoder.singleValueContainer().decode(Int.self)
        _ = try? decoder.singleValueContainer().decode(Double.self)
        _ = try? decoder.singleValueContainer().decode(String.self)
        _ = try? decoder.singleValueContainer().decode([AnyCodable].self)
        _ = try? decoder.singleValueContainer().decode([String: AnyCodable].self)
    }
}

struct TopologyConnection {
    let localNodeId: String
    let sendBackNodeId: String
}

// MARK: - Downloads

private struct TaggedNodeDownload: Decodable {
    let status: NodeDownloadStatus?

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let payloads = try container.decode([String: NodeDownloadPayload].self)
        guard let entry = payloads.first else {
            status = nil
            return
        }
        status = NodeDownloadStatus(statusKey: entry.key, payload: entry.value)
    }
}

struct NodeDownloadPayload: Decodable {
    let nodeId: String?
    let downloadProgress: DownloadProgress?
}

struct NodeDownloadStatus {
    let nodeId: String
    let progress: DownloadProgress?

    init?(statusKey: String, payload: NodeDownloadPayload) {
        guard let nodeId = payload.nodeId else { return nil }
        self.nodeId = nodeId
        self.progress = statusKey == "DownloadOngoing" ? payload.downloadProgress : nil
    }
}

struct DownloadProgress: Decodable {
    let totalBytes: ByteValue
    let downloadedBytes: ByteValue
    let speed: Double?
    let etaMs: Int64?
    let completedFiles: Int?
    let totalFiles: Int?
    let files: [String: FileDownloadProgress]?
}

struct ByteValue: Decodable {
    let inBytes: Int64
}

struct FileDownloadProgress: Decodable {
    let totalBytes: ByteValue
    let downloadedBytes: ByteValue
    let speed: Double?
    let etaMs: Int64?
}

// MARK: - Tasks

struct ClusterTask {
    enum Kind {
        case chatCompletion
    }

    let id: String
    let status: TaskStatus
    let instanceId: String?
    let kind: Kind
    let modelName: String?
    let promptPreview: String?
    let errorMessage: String?
    let parameters: ChatCompletionTaskParameters?

    var sortPriority: Int {
        switch status {
        case .running:
            return 0
        case .pending:
            return 1
        case .complete:
            return 2
        case .failed:
            return 3
        case .unknown:
            return 4
        }
    }
}

private struct TaggedTask: Decodable {
    let task: ClusterTask?

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let payloads = try container.decode([String: ClusterTaskPayload].self)
        guard let entry = payloads.first else {
            task = nil
            return
        }
        task = ClusterTask(kindKey: entry.key, payload: entry.value)
    }
}

struct ClusterTaskPayload: Decodable {
    let taskId: String?
    let taskStatus: TaskStatus?
    let instanceId: String?
    let commandId: String?
    let taskParams: ChatCompletionTaskParameters?
    let errorType: String?
    let errorMessage: String?
}

struct ChatCompletionTaskParameters: Decodable, Equatable {
    let model: String?
    let messages: [ChatCompletionMessage]?
    let maxTokens: Int?
    let stream: Bool?
    let temperature: Double?
    let topP: Double?

    private enum CodingKeys: String, CodingKey {
        case model
        case messages
        case maxTokens
        case stream
        case temperature
        case topP
    }

    func promptPreview() -> String? {
        guard let messages else { return nil }
        if let userMessage = messages.last(where: {
            $0.role?.lowercased() == "user" && ($0.content?.isEmpty == false)
        }) {
            return userMessage.content
        }
        return messages.last?.content
    }

}

struct ChatCompletionMessage: Decodable, Equatable {
    let role: String?
    let content: String?
}

extension ClusterTask {
    init?(kindKey: String, payload: ClusterTaskPayload) {
        guard let id = payload.taskId else { return nil }
        let status = payload.taskStatus ?? .unknown
        switch kindKey {
        case "ChatCompletion":
            self.init(
                id: id,
                status: status,
                instanceId: payload.instanceId,
                kind: .chatCompletion,
                modelName: payload.taskParams?.model,
                promptPreview: payload.taskParams?.promptPreview(),
                errorMessage: payload.errorMessage,
                parameters: payload.taskParams
            )
        default:
            return nil
        }
    }
}

enum TaskStatus: String, Decodable {
    case pending = "Pending"
    case running = "Running"
    case complete = "Complete"
    case failed = "Failed"
    case unknown

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        self = TaskStatus(rawValue: value) ?? .unknown
    }

    var displayLabel: String {
        switch self {
        case .pending, .running, .complete, .failed:
            return rawValue
        case .unknown:
            return "Unknown"
        }
    }
}

// MARK: - Derived summaries

struct ClusterOverview {
    let totalRam: Double
    let usedRam: Double
    let nodeCount: Int
    let instanceCount: Int
}

struct NodeSummary: Identifiable {
    let id: String
    let friendlyName: String
    let model: String
    let usedRamGB: Double
    let totalRamGB: Double
    let gpuUsagePercent: Double
    let temperatureCelsius: Double
}

struct InstanceSummary: Identifiable {
    let id: String
    let modelId: String
    let nodeCount: Int
    let statusText: String
}

extension ClusterState {
    func overview() -> ClusterOverview {
        var total: Double = 0
        var available: Double = 0
        for profile in nodeProfiles.values {
            if let totalBytes = profile.memory?.ramTotal?.inBytes {
                total += Double(totalBytes)
            }
            if let availableBytes = profile.memory?.ramAvailable?.inBytes {
                available += Double(availableBytes)
            }
        }
        let totalGB = total / 1_073_741_824.0
        let usedGB = max(total - available, 0) / 1_073_741_824.0
        return ClusterOverview(
            totalRam: totalGB,
            usedRam: usedGB,
            nodeCount: nodeProfiles.count,
            instanceCount: instances.count
        )
    }

    func availableModels() -> [ModelOption] { [] }
}
