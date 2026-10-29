import Foundation

extension ClusterStateService {
    /// Whether a model has a part running on this Mac, which restarting exo
    /// would stop. Nil while that isn't known, for example before exo answers.
    var isRunningModelsOnThisMac: Bool? {
        guard let snapshot = latestSnapshot, let localNodeId else { return nil }
        return Self.modelsRun(on: localNodeId, in: snapshot.instances.values)
    }

    nonisolated static func modelsRun<Instances: Sequence>(
        on nodeId: String, in instances: Instances
    ) -> Bool
    where Instances.Element == ClusterInstance {
        instances.contains { $0.shardAssignments.nodeToRunner[nodeId] != nil }
    }
}
