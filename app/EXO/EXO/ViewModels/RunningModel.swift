import Foundation

/// A model the cluster is running, as the menu bar shows it: a display name
/// and the same status words as the dashboard.
struct RunningModel: Identifiable, Equatable {
    /// The glossary's model statuses.
    enum Status: Equatable {
        case downloading(percent: Int)
        case starting
        case ready
        case failed
        case stopping

        var label: String {
            switch self {
            case .downloading(let percent): return "Downloading \(percent)%"
            case .starting: return "Starting"
            case .ready: return "Ready"
            case .failed: return "Failed"
            case .stopping: return "Stopping"
            }
        }
    }

    let id: String
    let displayName: String
    /// The model's repository id, e.g. mlx-community/Llama-3.2-3B-Instruct-4bit.
    let repoId: String
    let status: Status
    /// Whether this Mac runs part of it, so quitting EXO stops it.
    let runsOnThisMac: Bool
}

extension RunningModel.Status {
    /// Worked out like the dashboard's instance status: a failed runner wins,
    /// then stopping, then an ongoing download, then ready once every runner
    /// is ready; anything before that is starting.
    init(runnerStatuses: [String], downloadFraction: Double?) {
        let statuses = Set(runnerStatuses)
        if statuses.contains("RunnerFailed") {
            self = .failed
        } else if !statuses.isDisjoint(with: ["RunnerShutdown", "RunnerShuttingDown"]) {
            self = .stopping
        } else if let downloadFraction {
            self = .downloading(percent: Int((downloadFraction * 100).rounded(.down)))
        } else if !statuses.isEmpty,
            statuses.isSubset(of: ["RunnerReady", "RunnerRunning", "RunnerLoaded"])
        {
            self = .ready
        } else {
            self = .starting
        }
    }
}

extension ClusterState {
    /// The running models, by display name. `displayNames` maps repo ids to
    /// the names the dashboard shows (the model card's base model).
    func runningModels(localNodeId: String?, displayNames: [String: String]) -> [RunningModel] {
        instances.map { instanceId, instance in
            let assignments = instance.shardAssignments
            let nodeIds = Array(assignments.nodeToRunner.keys)
            let runnerStatuses = assignments.nodeToRunner.values.compactMap {
                runners[$0]?.status
            }
            return RunningModel(
                id: instanceId,
                displayName: displayNames[assignments.modelId]
                    ?? RunningModel.fallbackName(forRepoId: assignments.modelId),
                repoId: assignments.modelId,
                status: RunningModel.Status(
                    runnerStatuses: runnerStatuses,
                    downloadFraction: downloadFraction(of: assignments.modelId, on: nodeIds)),
                runsOnThisMac: localNodeId.map(nodeIds.contains) ?? false
            )
        }
        .sorted {
            ($0.displayName, $0.id) < ($1.displayName, $1.id)
        }
    }

    /// How much of `modelId` the given nodes have downloaded, or nil if none
    /// of them is downloading it.
    private func downloadFraction(of modelId: String, on nodeIds: [String]) -> Double? {
        var downloaded: Int64 = 0
        var total: Int64 = 0
        var isDownloading = false
        for nodeId in nodeIds {
            for download in downloads[nodeId] ?? [] where download.modelId == modelId {
                guard let progress = download.progress else { continue }
                isDownloading = true
                downloaded += progress.downloadedBytes.inBytes
                total += progress.totalBytes.inBytes
            }
        }
        guard isDownloading else { return nil }
        guard total > 0 else { return 0 }
        return min(max(Double(downloaded) / Double(total), 0), 1)
    }
}

extension RunningModel {
    /// The repo id without its organisation, for a model the app has no card
    /// for: "mlx-community/Llama-3.2-3B-Instruct-4bit" becomes
    /// "Llama-3.2-3B-Instruct-4bit".
    static func fallbackName(forRepoId repoId: String) -> String {
        repoId.split(separator: "/").last.map(String.init) ?? repoId
    }
}
