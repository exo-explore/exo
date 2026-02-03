import SwiftUI

struct TopologyMiniView: View {
    let topology: TopologyViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Topology")
                .font(.caption)
                .foregroundColor(.secondary)
            GeometryReader { geo in
                ZStack {
                    connectionLines(in: geo.size)
                    let positions = positionedNodes(in: geo.size)
                    ForEach(Array(positions.enumerated()), id: \.element.node.id) { _, positioned in
                        NodeGlyphView(node: positioned.node, isCurrent: positioned.isCurrent)
                            .position(positioned.point)
                    }
                }
            }
            .frame(height: heightForNodes())
        }
    }

    private func positionedNodes(in size: CGSize) -> [PositionedNode] {
        let nodes = orderedNodesForLayout()
        guard !nodes.isEmpty else { return [] }
        var result: [PositionedNode] = []
        let glyphHeight: CGFloat = 70
        let rootPoint = CGPoint(x: size.width / 2, y: glyphHeight / 2 + 10)
        result.append(
            PositionedNode(
                node: nodes[0],
                point: rootPoint,
                isCurrent: nodes[0].id == topology.currentNodeId
            )
        )
        guard nodes.count > 1 else { return result }
        let childCount = nodes.count - 1
        // Larger radius to reduce overlap when several nodes exist
        let minDimension = min(size.width, size.height)
        let radius = max(120, minDimension * 0.42)
        let startAngle = Double.pi * 0.75
        let endAngle = Double.pi * 0.25
        let step = childCount == 1 ? 0 : (startAngle - endAngle) / Double(childCount - 1)
        for (index, node) in nodes.dropFirst().enumerated() {
            let angle = startAngle - step * Double(index)
            let x = size.width / 2 + radius * CGFloat(cos(angle))
            let y = rootPoint.y + radius * CGFloat(sin(angle)) + glyphHeight / 2
            result.append(
                PositionedNode(
                    node: node,
                    point: CGPoint(x: x, y: y),
                    isCurrent: node.id == topology.currentNodeId
                )
            )
        }
        return result
    }

    private func orderedNodesForLayout() -> [NodeViewModel] {
        guard let currentId = topology.currentNodeId else {
            return topology.nodes
        }
        guard let currentIndex = topology.nodes.firstIndex(where: { $0.id == currentId }) else {
            return topology.nodes
        }
        if currentIndex == 0 {
            return topology.nodes
        }
        var reordered = topology.nodes
        let current = reordered.remove(at: currentIndex)
        reordered.insert(current, at: 0)
        return reordered
    }

    private func connectionLines(in size: CGSize) -> some View {
        let positions = positionedNodes(in: size)
        let positionById = Dictionary(
            uniqueKeysWithValues: positions.map { ($0.node.id, $0.point) })
        return Canvas { context, _ in
            guard !topology.edges.isEmpty else { return }
            let nodeRadius: CGFloat = 32
            let arrowLength: CGFloat = 10
            let arrowSpread: CGFloat = .pi / 7
            for edge in topology.edges {
                guard let start = positionById[edge.sourceId], let end = positionById[edge.targetId]
                else { continue }
                let dx = end.x - start.x
                let dy = end.y - start.y
                let distance = max(CGFloat(hypot(dx, dy)), 1)
                let ux = dx / distance
                let uy = dy / distance
                let adjustedStart = CGPoint(
                    x: start.x + ux * nodeRadius, y: start.y + uy * nodeRadius)
                let adjustedEnd = CGPoint(x: end.x - ux * nodeRadius, y: end.y - uy * nodeRadius)

                var linePath = Path()
                linePath.move(to: adjustedStart)
                linePath.addLine(to: adjustedEnd)
                context.stroke(
                    linePath,
                    with: .color(.secondary.opacity(0.3)),
                    style: StrokeStyle(lineWidth: 1, dash: [4, 4])
                )

                let angle = atan2(uy, ux)
                let tip = adjustedEnd
                let leftWing = CGPoint(
                    x: tip.x - arrowLength * cos(angle - arrowSpread),
                    y: tip.y - arrowLength * sin(angle - arrowSpread)
                )
                let rightWing = CGPoint(
                    x: tip.x - arrowLength * cos(angle + arrowSpread),
                    y: tip.y - arrowLength * sin(angle + arrowSpread)
                )
                var arrowPath = Path()
                arrowPath.move(to: tip)
                arrowPath.addLine(to: leftWing)
                arrowPath.move(to: tip)
                arrowPath.addLine(to: rightWing)
                context.stroke(
                    arrowPath,
                    with: .color(.secondary.opacity(0.5)),
                    style: StrokeStyle(lineWidth: 1)
                )
            }
        }
    }

    private func heightForNodes() -> CGFloat {
        switch topology.nodes.count {
        case 0...1:
            return 130
        case 2...3:
            return 200
        default:
            return 240
        }
    }

    private struct PositionedNode {
        let node: NodeViewModel
        let point: CGPoint
        let isCurrent: Bool
    }
}

private struct NodeGlyphView: View {
    let node: NodeViewModel
    let isCurrent: Bool

    var body: some View {
        VStack(spacing: 2) {
            Image(systemName: node.deviceIconName)
                .font(.subheadline)
            Text(node.friendlyName)
                .font(.caption2)
                .lineLimit(1)
                .foregroundColor(isCurrent ? Color(nsColor: .systemBlue) : .primary)
            Text(node.memoryLabel)
                .font(.caption2)
            HStack(spacing: 3) {
                Text(node.gpuUsageLabel)
                Text(node.temperatureLabel)
            }
            .foregroundColor(.secondary)
            .font(.caption2)
        }
        .padding(.vertical, 3)
        .frame(width: 95)
    }
}
