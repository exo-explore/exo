import Foundation

/// A device in the cluster, as the menu bar shows it: its name and how much
/// of its memory is in use.
struct ClusterDevice: Identifiable, Equatable {
    let id: String
    let name: String
    let isThisMac: Bool
    let usedMemoryBytes: Int64
    let totalMemoryBytes: Int64

    /// "41 of 192 GB", or nil before the device has reported its memory.
    var memoryInUse: String? {
        guard totalMemoryBytes > 0 else { return nil }
        return ClusterDevice.memoryInUse(used: usedMemoryBytes, total: totalMemoryBytes)
    }

    /// "21 of 64 GB": whole gigabytes (GiB, as the dashboard counts them).
    static func memoryInUse(used: Int64, total: Int64) -> String {
        let gigabyte = 1_073_741_824.0
        let usedGB = Int((Double(max(used, 0)) / gigabyte).rounded())
        let totalGB = Int((Double(total) / gigabyte).rounded())
        return "\(usedGB) of \(totalGB) GB"
    }
}

extension ClusterState {
    /// The cluster's devices, this Mac first, then the others by name.
    func devices(localNodeId: String?) -> [ClusterDevice] {
        let nodeIds = Set(nodeIdentities.keys).union(nodeMemory.keys)
        return nodeIds.map { nodeId in
            let identity = nodeIdentities[nodeId]
            let memory = nodeMemory[nodeId]
            let total = memory?.ramTotal?.inBytes ?? 0
            let available = memory?.ramAvailable?.inBytes ?? 0
            let name =
                [identity?.friendlyName, identity?.modelId]
                .compactMap { $0 }
                .first { !$0.isEmpty && $0 != "Unknown" } ?? "Unknown device"
            return ClusterDevice(
                id: nodeId,
                name: name,
                isThisMac: nodeId == localNodeId,
                usedMemoryBytes: max(total - available, 0),
                totalMemoryBytes: total
            )
        }
        .sorted {
            if $0.isThisMac != $1.isThisMac { return $0.isThisMac }
            return ($0.name, $0.id) < ($1.name, $1.id)
        }
    }
}
