import SwiftUI

struct NodeRowView: View {
    let node: NodeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                VStack(alignment: .leading) {
                    Text(node.friendlyName)
                        .font(.subheadline)
                    Text(node.memoryLabel)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing) {
                    Text("\(node.gpuUsagePercent, specifier: "%.0f")% GPU")
                        .font(.caption)
                    Text(node.temperatureLabel)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            ProgressView(value: node.memoryProgress)
                .progressViewStyle(.linear)
        }
        .padding(.vertical, 4)
    }
}
