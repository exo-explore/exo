import SwiftUI

struct NodeDetailView: View {
    let node: NodeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(node.friendlyName)
                .font(.headline)
            Text(node.model)
                .font(.caption)
                .foregroundColor(.secondary)
            Divider()
            metricRow(label: "Memory", value: node.memoryLabel)
            ProgressView(value: node.memoryProgress)
            metricRow(label: "CPU Usage", value: node.cpuUsageLabel)
            metricRow(label: "GPU Usage", value: node.gpuUsageLabel)
            metricRow(label: "Temperature", value: node.temperatureLabel)
            metricRow(label: "Power", value: node.powerLabel)
        }
        .padding()
    }

    private func metricRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.subheadline)
        }
    }
}
