import SwiftUI

struct SettingsView: View {
    @Environment(ClusterService.self) private var clusterService
    @Environment(DiscoveryService.self) private var discoveryService
    @Environment(LocalInferenceService.self) private var localInferenceService
    @Environment(\.dismiss) private var dismiss
    @State private var host: String = ""
    @State private var port: String = "52415"

    var body: some View {
        NavigationStack {
            Form {
                localModelSection
                nearbyClustersSection
                connectionSection
                statusSection
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var localModelSection: some View {
        Section {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(localInferenceService.defaultModelId)
                        .font(.subheadline)
                        .fontWeight(.medium)

                    Text(localModelStatusText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                localModelActionButton
            }

            if case .downloading(let progress) = localInferenceService.modelState {
                ProgressView(value: progress)
                    .tint(.blue)
            }
        } header: {
            Text("Local Model")
        } footer: {
            Text("When disconnected from a cluster, messages are processed on-device using this model.")
        }
    }

    private var localModelStatusText: String {
        switch localInferenceService.modelState {
        case .notDownloaded: "Not downloaded"
        case .downloading(let progress): "Downloading \(Int(progress * 100))%..."
        case .downloaded: "Downloaded — not loaded"
        case .loading: "Loading into memory..."
        case .ready: "Ready"
        case .generating: "Generating..."
        case .error(let message): "Error: \(message)"
        }
    }

    @ViewBuilder
    private var localModelActionButton: some View {
        switch localInferenceService.modelState {
        case .notDownloaded:
            Button("Download") {
                Task { await localInferenceService.prepareModel() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        case .downloading:
            ProgressView()
                .controlSize(.small)
        case .downloaded:
            Button("Load") {
                Task { await localInferenceService.prepareModel() }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        case .loading:
            ProgressView()
                .controlSize(.small)
        case .ready, .generating:
            Button("Unload") {
                localInferenceService.unloadModel()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        case .error:
            Button("Retry") {
                Task { await localInferenceService.prepareModel() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .tint(.red)
        }
    }

    private var nearbyClustersSection: some View {
        Section {
            if discoveryService.discoveredClusters.isEmpty {
                if discoveryService.isSearching {
                    HStack {
                        ProgressView()
                            .padding(.trailing, 8)
                        Text("Searching for clusters...")
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text("No clusters found")
                        .foregroundStyle(.secondary)
                }
            } else {
                ForEach(discoveryService.discoveredClusters) { cluster in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(cluster.name)
                                .font(.body)
                        }
                        Spacer()
                        Button("Connect") {
                            Task {
                                await clusterService.connectToDiscoveredCluster(
                                    cluster, using: discoveryService
                                )
                                if clusterService.isConnected {
                                    dismiss()
                                }
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                }
            }
        } header: {
            Text("Nearby Clusters")
        }
    }

    private var connectionSection: some View {
        Section("Manual Connection") {
            TextField("IP Address (e.g. 192.168.1.42)", text: $host)
                .keyboardType(.decimalPad)
                .textContentType(.URL)
                .autocorrectionDisabled()

            TextField("Port", text: $port)
                .keyboardType(.numberPad)

            Button(clusterService.isConnected ? "Reconnect" : "Connect") {
                Task {
                    let portNum = Int(port) ?? ConnectionInfo.defaultPort
                    let info = ConnectionInfo(host: host, port: portNum, nodeId: nil)
                    await clusterService.connect(to: info)
                    if clusterService.isConnected {
                        dismiss()
                    }
                }
            }
            .disabled(host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    private var statusSection: some View {
        Section("Status") {
            if let connection = clusterService.currentConnection {
                LabeledContent("Host", value: connection.host)
                LabeledContent("Port", value: "\(connection.port)")
                if let nodeId = connection.nodeId {
                    LabeledContent("Node ID", value: String(nodeId.prefix(12)) + "...")
                }
                LabeledContent("Models", value: "\(clusterService.availableModels.count)")

                Button("Disconnect", role: .destructive) {
                    clusterService.disconnect()
                }
            } else {
                if let error = clusterService.lastError {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                } else {
                    Text("Not connected")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}
