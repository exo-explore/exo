import Foundation

struct NodeViewModel: Identifiable, Equatable {
    let id: String
    let friendlyName: String
    let model: String
    let usedRamGB: Double
    let totalRamGB: Double
    let gpuUsagePercent: Double
    let cpuUsagePercent: Double
    let temperatureCelsius: Double
    let systenPowerWatts: Double

    var memoryProgress: Double {
        guard totalRamGB > 0 else { return 0 }
        return min(max(usedRamGB / totalRamGB, 0), 1)
    }

    var memoryLabel: String {
        String(format: "%.1f / %.1f GB", usedRamGB, totalRamGB)
    }

    var temperatureLabel: String {
        String(format: "%.0f°C", temperatureCelsius)
    }

    var powerLabel: String {
        systenPowerWatts > 0 ? String(format: "%.0fW", systenPowerWatts) : "—"
    }

    var cpuUsageLabel: String {
        String(format: "%.0f%%", cpuUsagePercent)
    }

    var gpuUsageLabel: String {
        String(format: "%.0f%%", gpuUsagePercent)
    }

    var deviceIconName: String {
        let lower = model.lowercased()
        if lower.contains("studio") {
            return "macstudio"
        }
        if lower.contains("mini") {
            return "macmini"
        }
        return "macbook"
    }
}

extension ClusterState {
    func nodeViewModels() -> [NodeViewModel] {
        nodeProfiles.map { entry in
            let profile = entry.value
            let friendly = profile.friendlyName ?? profile.modelId ?? entry.key
            let model = profile.modelId ?? "Unknown"
            let totalBytes = Double(profile.memory?.ramTotal?.inBytes ?? 0)
            let availableBytes = Double(profile.memory?.ramAvailable?.inBytes ?? 0)
            let usedBytes = max(totalBytes - availableBytes, 0)
            return NodeViewModel(
                id: entry.key,
                friendlyName: friendly,
                model: model,
                usedRamGB: usedBytes / 1_073_741_824.0,
                totalRamGB: totalBytes / 1_073_741_824.0,
                gpuUsagePercent: (profile.system?.gpuUsage ?? 0) * 100,
                cpuUsagePercent: (profile.system?.pcpuUsage ?? 0) * 100,
                temperatureCelsius: profile.system?.temp ?? 0,
                systenPowerWatts: profile.system?.sysPower ?? 0
            )
        }
        .sorted { $0.friendlyName < $1.friendlyName }
    }
}

struct TopologyEdgeViewModel: Hashable {
    let sourceId: String
    let targetId: String
}

struct TopologyViewModel {
    let nodes: [NodeViewModel]
    let edges: [TopologyEdgeViewModel]
    let currentNodeId: String?
}

extension ClusterState {
    func topologyViewModel(localNodeId: String?) -> TopologyViewModel? {
        let topologyNodeIds = Set(topology?.nodes.map(\.nodeId) ?? [])
        let allNodes = nodeViewModels().filter {
            topologyNodeIds.isEmpty || topologyNodeIds.contains($0.id)
        }
        guard !allNodes.isEmpty else { return nil }

        let nodesById = Dictionary(uniqueKeysWithValues: allNodes.map { ($0.id, $0) })
        var orderedNodes: [NodeViewModel] = []
        if let topologyNodes = topology?.nodes {
            for topoNode in topologyNodes {
                if let viewModel = nodesById[topoNode.nodeId] {
                    orderedNodes.append(viewModel)
                }
            }
            let seenIds = Set(orderedNodes.map(\.id))
            let remaining = allNodes.filter { !seenIds.contains($0.id) }
            orderedNodes.append(contentsOf: remaining)
        } else {
            orderedNodes = allNodes
        }

        // Rotate so the local node (from /node_id API) is first
        if let localId = localNodeId,
            let index = orderedNodes.firstIndex(where: { $0.id == localId })
        {
            orderedNodes = Array(orderedNodes[index...]) + Array(orderedNodes[..<index])
        }

        let nodeIds = Set(orderedNodes.map(\.id))
        let edgesArray: [TopologyEdgeViewModel] =
            topology?.connections?.compactMap { connection in
                guard nodeIds.contains(connection.localNodeId),
                    nodeIds.contains(connection.sendBackNodeId)
                else { return nil }
                return TopologyEdgeViewModel(
                    sourceId: connection.localNodeId, targetId: connection.sendBackNodeId)
            } ?? []
        let edges = Set(edgesArray)

        return TopologyViewModel(
            nodes: orderedNodes, edges: Array(edges), currentNodeId: localNodeId)
    }
}
