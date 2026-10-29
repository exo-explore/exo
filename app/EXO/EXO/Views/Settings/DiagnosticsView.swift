import AppKit
import SwiftUI

/// The network state of this Mac and the cluster, for diagnosing problems.
/// Each state is a word with a symbol, not a colour.
struct DiagnosticsSummary: Equatable {
    struct Item: Equatable {
        enum Mark: Equatable {
            case good
            case warning
            case neutral
            case unknown

            var systemImage: String {
                switch self {
                case .good: return "checkmark.circle"
                case .warning: return "exclamationmark.triangle"
                case .neutral: return "minus.circle"
                case .unknown: return "questionmark.circle"
                }
            }
        }

        let label: String
        let value: String
        let mark: Mark
    }

    struct Category: Equatable {
        let title: String
        let items: [Item]
    }

    let categories: [Category]

    static func make(
        network: NetworkStatus, cluster: ClusterState?, localNodeId: String?
    ) -> DiagnosticsSummary {
        var thisMac: [Item] = [
            bridgeItem("Thunderbolt Bridge", network.thunderboltBridgeState)
        ]
        if let localNodeId, let rdma = cluster?.nodeRdmaCtl[localNodeId] {
            thisMac.append(rdmaItem(enabled: rdma.enabled))
        }
        if !network.localRdmaDevices.isEmpty {
            thisMac.append(
                Item(
                    label: "RDMA devices",
                    value: network.localRdmaDevices.joined(separator: ", "),
                    mark: .neutral))
        }
        for port in network.localRdmaActivePorts {
            thisMac.append(
                Item(
                    label: "\(port.device) port \(port.port)", value: port.state, mark: .neutral))
        }

        // Interfaces with an address get a row each; the rest share one.
        var interfaces = network.interfaceStatuses.compactMap { status -> Item? in
            guard let address = status.ipAddress else { return nil }
            return Item(label: status.interfaceName, value: address, mark: .good)
        }
        let withoutAddress = network.interfaceStatuses.filter { $0.ipAddress == nil }
        if !withoutAddress.isEmpty {
            interfaces.append(
                Item(
                    label: "No address",
                    value: withoutAddress.map(\.interfaceName).joined(separator: ", "),
                    mark: .neutral))
        }

        var devices: [Item] = []
        if let cluster {
            let profiles = cluster.nodeProfiles
            let nodeIds = Set(cluster.nodeThunderboltBridge.keys)
                .union(cluster.nodeRdmaCtl.keys)
                .subtracting([localNodeId].compactMap { $0 })
            for nodeId in nodeIds.sorted() {
                let name = profiles[nodeId]?.friendlyName ?? String(nodeId.prefix(8))
                var parts: [String] = []
                var mark = Item.Mark.good
                if let bridge = cluster.nodeThunderboltBridge[nodeId] {
                    if !bridge.exists {
                        parts.append("no Thunderbolt Bridge")
                    } else if bridge.enabled {
                        parts.append("Thunderbolt Bridge on")
                        mark = .warning
                    } else {
                        parts.append("Thunderbolt Bridge off")
                    }
                }
                if let rdma = cluster.nodeRdmaCtl[nodeId] {
                    parts.append(rdma.enabled ? "RDMA on" : "RDMA off")
                }
                devices.append(Item(label: name, value: parts.joined(separator: ", "), mark: mark))
            }
        }

        var categories = [Category(title: "This Mac", items: thisMac)]
        categories.append(
            Category(
                title: "Network interfaces",
                items: interfaces.isEmpty
                    ? [Item(label: "Interfaces", value: "Unknown", mark: .unknown)] : interfaces))
        if !devices.isEmpty {
            categories.append(Category(title: "Other devices", items: devices))
        }
        return DiagnosticsSummary(categories: categories)
    }

    /// As plain text, for pasting into a bug report or a chat.
    var text: String {
        categories.map { category in
            ([category.title + ":"] + category.items.map { "  \($0.label): \($0.value)" })
                .joined(separator: "\n")
        }.joined(separator: "\n")
    }

    private static func bridgeItem(_ label: String, _ state: ThunderboltState?) -> Item {
        switch state {
        case .disabled:
            return Item(label: label, value: "Off", mark: .good)
        case .deleted:
            return Item(label: label, value: "Removed", mark: .good)
        case .enabled:
            return Item(label: label, value: "On", mark: .warning)
        case nil:
            return Item(label: label, value: "None", mark: .neutral)
        }
    }

    private static func rdmaItem(enabled: Bool) -> Item {
        Item(label: "RDMA", value: enabled ? "On" : "Off", mark: enabled ? .good : .neutral)
    }
}

/// Settings › Advanced › Diagnostics. Refreshes only while it is shown.
struct DiagnosticsView: View {
    @EnvironmentObject private var networkStatusService: NetworkStatusService
    @EnvironmentObject private var stateService: ClusterStateService

    private var summary: DiagnosticsSummary {
        DiagnosticsSummary.make(
            network: networkStatusService.status,
            cluster: stateService.latestSnapshot,
            localNodeId: stateService.localNodeId)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(summary.categories, id: \.title) { category in
                VStack(alignment: .leading, spacing: 4) {
                    Text(category.title)
                        .font(.subheadline.weight(.semibold))
                    ForEach(Array(category.items.enumerated()), id: \.offset) { _, item in
                        LabeledContent(item.label) {
                            Label(item.value, systemImage: item.mark.systemImage)
                                .textSelection(.enabled)
                        }
                    }
                }
            }
            HStack {
                Spacer()
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(summary.text, forType: .string)
                }
            }
        }
        .padding(.top, 4)
        .task {
            while !Task.isCancelled {
                await networkStatusService.refresh()
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }
}
