import Foundation

struct DownloadProgressViewModel: Equatable {
    let downloadedBytes: Int64
    let totalBytes: Int64
    let speedBytesPerSecond: Double
    let etaSeconds: Double?
    let completedFiles: Int
    let totalFiles: Int

    var fractionCompleted: Double {
        guard totalBytes > 0 else { return 0 }
        return Double(downloadedBytes) / Double(totalBytes)
    }

    var percentCompleted: Double {
        fractionCompleted * 100
    }

    var formattedProgress: String {
        let downloaded = formatBytes(downloadedBytes)
        let total = formatBytes(totalBytes)
        let percent = String(format: "%.1f", percentCompleted)
        return "\(downloaded)/\(total) (\(percent)%)"
    }

    var formattedSpeed: String {
        "\(formatBytes(Int64(speedBytesPerSecond)))/s"
    }

    var formattedETA: String? {
        guard let eta = etaSeconds, eta > 0 else { return nil }
        let minutes = Int(eta) / 60
        let seconds = Int(eta) % 60
        if minutes > 0 {
            return "ETA \(minutes)m \(seconds)s"
        }
        return "ETA \(seconds)s"
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_073_741_824.0
        let mb = Double(bytes) / 1_048_576.0
        if gb >= 1.0 {
            return String(format: "%.2f GB", gb)
        }
        return String(format: "%.0f MB", mb)
    }
}

struct InstanceViewModel: Identifiable, Equatable {
    enum State {
        case downloading
        case warmingUp
        case running
        case ready
        case waiting
        case failed
        case idle
        case unknown

        var label: String {
            switch self {
            case .downloading: return "Downloading"
            case .warmingUp: return "Warming Up"
            case .running: return "Running"
            case .ready: return "Ready"
            case .waiting: return "Waiting"
            case .failed: return "Failed"
            case .idle: return "Idle"
            case .unknown: return "Unknown"
            }
        }
    }

    let id: String
    let modelName: String
    let sharding: String?
    let nodeNames: [String]
    let state: State
    let chatTasks: [InstanceTaskViewModel]
    let downloadProgress: DownloadProgressViewModel?

    var nodeSummary: String {
        guard !nodeNames.isEmpty else { return "0 nodes" }
        if nodeNames.count == 1 {
            return nodeNames[0]
        }
        if nodeNames.count == 2 {
            return nodeNames.joined(separator: ", ")
        }
        let others = nodeNames.count - 1
        return "\(nodeNames.first ?? "") +\(others)"
    }
}

extension ClusterState {
    func instanceViewModels() -> [InstanceViewModel] {
        let chatTasksByInstance = Dictionary(
            grouping: tasks.values.filter { $0.kind == .chatCompletion && $0.instanceId != nil },
            by: { $0.instanceId! }
        )

        return instances.map { entry in
            let instance = entry.value
            let modelName = instance.shardAssignments.modelId
            let nodeToRunner = instance.shardAssignments.nodeToRunner
            let nodeIds = Array(nodeToRunner.keys)
            let runnerIds = Array(nodeToRunner.values)
            let nodeNames = nodeIds.compactMap {
                nodeProfiles[$0]?.friendlyName ?? nodeProfiles[$0]?.modelId ?? $0
            }
            let statuses = runnerIds.compactMap { runners[$0]?.status.lowercased() }
            let downloadProgress = aggregateDownloadProgress(for: nodeIds)
            let state = InstanceViewModel.State(
                statuses: statuses, hasActiveDownload: downloadProgress != nil)
            let chatTasks = (chatTasksByInstance[entry.key] ?? [])
                .sorted(by: { $0.sortPriority < $1.sortPriority })
                .map { InstanceTaskViewModel(task: $0) }
            return InstanceViewModel(
                id: entry.key,
                modelName: modelName,
                sharding: InstanceViewModel.friendlyShardingName(for: instance.sharding),
                nodeNames: nodeNames,
                state: state,
                chatTasks: chatTasks,
                downloadProgress: downloadProgress
            )
        }
        .sorted { $0.modelName < $1.modelName }
    }

    private func aggregateDownloadProgress(for nodeIds: [String]) -> DownloadProgressViewModel? {
        var totalDownloaded: Int64 = 0
        var totalSize: Int64 = 0
        var totalSpeed: Double = 0
        var maxEtaMs: Int64 = 0
        var totalCompletedFiles = 0
        var totalFileCount = 0
        var hasActiveDownload = false

        for nodeId in nodeIds {
            guard let nodeDownloads = downloads[nodeId] else { continue }
            for download in nodeDownloads {
                guard let progress = download.progress else { continue }
                hasActiveDownload = true
                totalDownloaded += progress.downloadedBytes.inBytes
                totalSize += progress.totalBytes.inBytes
                totalSpeed += progress.speed ?? 0
                if let eta = progress.etaMs {
                    maxEtaMs = max(maxEtaMs, eta)
                }
                totalCompletedFiles += progress.completedFiles ?? 0
                totalFileCount += progress.totalFiles ?? 0
            }
        }

        guard hasActiveDownload else { return nil }

        return DownloadProgressViewModel(
            downloadedBytes: totalDownloaded,
            totalBytes: totalSize,
            speedBytesPerSecond: totalSpeed,
            etaSeconds: maxEtaMs > 0 ? Double(maxEtaMs) / 1000.0 : nil,
            completedFiles: totalCompletedFiles,
            totalFiles: totalFileCount
        )
    }
}

extension InstanceViewModel.State {
    fileprivate init(statuses: [String], hasActiveDownload: Bool = false) {
        if statuses.contains(where: { $0.contains("failed") }) {
            self = .failed
        } else if hasActiveDownload || statuses.contains(where: { $0.contains("downloading") }) {
            self = .downloading
        } else if statuses.contains(where: { $0.contains("warming") }) {
            self = .warmingUp
        } else if statuses.contains(where: { $0.contains("running") }) {
            self = .running
        } else if statuses.contains(where: { $0.contains("ready") || $0.contains("loaded") }) {
            self = .ready
        } else if statuses.contains(where: { $0.contains("waiting") }) {
            self = .waiting
        } else if statuses.isEmpty {
            self = .idle
        } else {
            self = .unknown
        }
    }
}

extension InstanceViewModel {
    static func friendlyShardingName(for raw: String?) -> String? {
        guard let raw else { return nil }
        switch raw.lowercased() {
        case "mlxringinstance", "mlxring":
            return "MLX Ring"
        case "mlxibvinstance", "mlxibv":
            return "MLX RDMA"
        default:
            return raw
        }
    }
}

struct InstanceTaskViewModel: Identifiable, Equatable {
    enum Kind {
        case chatCompletion
    }

    let id: String
    let kind: Kind
    let status: TaskStatus
    let modelName: String?
    let promptPreview: String?
    let errorMessage: String?
    let subtitle: String?
    let parameters: ChatCompletionTaskParameters?

    var title: String {
        switch kind {
        case .chatCompletion:
            return "Chat Completion"
        }
    }

    var detailText: String? {
        if let errorMessage, !errorMessage.isEmpty {
            return errorMessage
        }
        return promptPreview
    }

}

extension InstanceTaskViewModel {
    init(task: ClusterTask) {
        self.id = task.id
        self.kind = .chatCompletion
        self.status = task.status
        self.modelName = task.modelName
        self.promptPreview = task.promptPreview
        self.errorMessage = task.errorMessage
        self.subtitle = task.modelName
        self.parameters = task.parameters
    }
}
