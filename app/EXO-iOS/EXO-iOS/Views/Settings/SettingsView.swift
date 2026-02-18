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
            .scrollContentBackground(.hidden)
            .background(Color.exoBlack)
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .font(.exoSubheadline)
                        .foregroundStyle(Color.exoYellow)
                }
            }
        }
    }

    // MARK: - Section Headers

    private func sectionHeader(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.exoMono(10, weight: .semibold))
            .tracking(2)
            .foregroundStyle(Color.exoYellow)
    }

    // MARK: - Local Model

    private var localModelSection: some View {
        Section {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(localInferenceService.defaultModelId)
                        .font(.exoSubheadline)
                        .foregroundStyle(Color.exoForeground)

                    Text(localModelStatusText)
                        .font(.exoCaption)
                        .foregroundStyle(Color.exoLightGray)
                }

                Spacer()

                localModelActionButton
            }
            .listRowBackground(Color.exoDarkGray)

            if case .downloading(let progress) = localInferenceService.modelState {
                ProgressView(value: progress)
                    .tint(Color.exoYellow)
                    .listRowBackground(Color.exoDarkGray)
            }
        } header: {
            sectionHeader("Local Model")
        } footer: {
            Text(
                "When disconnected from a cluster, messages are processed on-device using this model."
            )
            .font(.exoCaption)
            .foregroundStyle(Color.exoLightGray.opacity(0.7))
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
            exoButton("Download") {
                Task { await localInferenceService.prepareModel() }
            }
        case .downloading:
            ProgressView()
                .controlSize(.small)
                .tint(Color.exoYellow)
        case .downloaded:
            exoButton("Load") {
                Task { await localInferenceService.prepareModel() }
            }
        case .loading:
            ProgressView()
                .controlSize(.small)
                .tint(Color.exoYellow)
        case .ready, .generating:
            exoButton("Unload") {
                localInferenceService.unloadModel()
            }
        case .error:
            exoButton("Retry", destructive: true) {
                Task { await localInferenceService.prepareModel() }
            }
        }
    }

    private func exoButton(_ title: String, destructive: Bool = false, action: @escaping () -> Void)
        -> some View
    {
        let borderColor = destructive ? Color.exoDestructive : Color.exoYellow
        return Button(action: action) {
            Text(title.uppercased())
                .font(.exoMono(11, weight: .semibold))
                .tracking(1)
                .foregroundStyle(borderColor)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(borderColor, lineWidth: 1)
                )
        }
    }

    // MARK: - Nearby Clusters

    private var nearbyClustersSection: some View {
        Section {
            if discoveryService.discoveredClusters.isEmpty {
                if discoveryService.isSearching {
                    HStack {
                        ProgressView()
                            .tint(Color.exoYellow)
                            .padding(.trailing, 8)
                        Text("Searching for clusters...")
                            .font(.exoBody)
                            .foregroundStyle(Color.exoLightGray)
                    }
                    .listRowBackground(Color.exoDarkGray)
                } else {
                    Text("No clusters found")
                        .font(.exoBody)
                        .foregroundStyle(Color.exoLightGray)
                        .listRowBackground(Color.exoDarkGray)
                }
            } else {
                ForEach(discoveryService.discoveredClusters) { cluster in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(cluster.name)
                                .font(.exoBody)
                                .foregroundStyle(Color.exoForeground)
                        }
                        Spacer()
                        exoButton("Connect") {
                            Task {
                                await clusterService.connectToDiscoveredCluster(
                                    cluster, using: discoveryService
                                )
                                if clusterService.isConnected {
                                    dismiss()
                                }
                            }
                        }
                    }
                    .listRowBackground(Color.exoDarkGray)
                }
            }
        } header: {
            sectionHeader("Nearby Clusters")
        }
    }

    // MARK: - Manual Connection

    private var connectionSection: some View {
        Section {
            TextField("IP Address (e.g. 192.168.1.42)", text: $host)
                .font(.exoBody)
                .keyboardType(.decimalPad)
                .textContentType(.URL)
                .autocorrectionDisabled()
                .foregroundStyle(Color.exoForeground)
                .listRowBackground(Color.exoDarkGray)

            TextField("Port", text: $port)
                .font(.exoBody)
                .keyboardType(.numberPad)
                .foregroundStyle(Color.exoForeground)
                .listRowBackground(Color.exoDarkGray)

            Button {
                Task {
                    let portNum = Int(port) ?? ConnectionInfo.defaultPort
                    let info = ConnectionInfo(host: host, port: portNum, nodeId: nil)
                    await clusterService.connect(to: info)
                    if clusterService.isConnected {
                        dismiss()
                    }
                }
            } label: {
                Text(clusterService.isConnected ? "RECONNECT" : "CONNECT")
                    .font(.exoMono(13, weight: .semibold))
                    .tracking(1.5)
                    .foregroundStyle(
                        host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            ? Color.exoLightGray : Color.exoYellow
                    )
                    .frame(maxWidth: .infinity)
            }
            .disabled(host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .listRowBackground(Color.exoDarkGray)
        } header: {
            sectionHeader("Manual Connection")
        }
    }

    // MARK: - Status

    private var statusSection: some View {
        Section {
            if let connection = clusterService.currentConnection {
                LabeledContent {
                    Text(connection.host)
                        .font(.exoCaption)
                        .foregroundStyle(Color.exoForeground)
                } label: {
                    Text("Host")
                        .font(.exoCaption)
                        .foregroundStyle(Color.exoLightGray)
                }
                .listRowBackground(Color.exoDarkGray)

                LabeledContent {
                    Text("\(connection.port)")
                        .font(.exoCaption)
                        .foregroundStyle(Color.exoForeground)
                } label: {
                    Text("Port")
                        .font(.exoCaption)
                        .foregroundStyle(Color.exoLightGray)
                }
                .listRowBackground(Color.exoDarkGray)

                if let nodeId = connection.nodeId {
                    LabeledContent {
                        Text(String(nodeId.prefix(12)) + "...")
                            .font(.exoCaption)
                            .foregroundStyle(Color.exoForeground)
                    } label: {
                        Text("Node ID")
                            .font(.exoCaption)
                            .foregroundStyle(Color.exoLightGray)
                    }
                    .listRowBackground(Color.exoDarkGray)
                }

                LabeledContent {
                    Text("\(clusterService.availableModels.count)")
                        .font(.exoCaption)
                        .foregroundStyle(Color.exoForeground)
                } label: {
                    Text("Models")
                        .font(.exoCaption)
                        .foregroundStyle(Color.exoLightGray)
                }
                .listRowBackground(Color.exoDarkGray)

                Button(role: .destructive) {
                    clusterService.disconnect()
                } label: {
                    Text("DISCONNECT")
                        .font(.exoMono(13, weight: .semibold))
                        .tracking(1.5)
                        .foregroundStyle(Color.exoDestructive)
                        .frame(maxWidth: .infinity)
                }
                .listRowBackground(Color.exoDarkGray)
            } else {
                if let error = clusterService.lastError {
                    Label {
                        Text(error)
                            .font(.exoCaption)
                    } icon: {
                        Image(systemName: "exclamationmark.triangle")
                    }
                    .foregroundStyle(Color.exoDestructive)
                    .listRowBackground(Color.exoDarkGray)
                } else {
                    Text("Not connected")
                        .font(.exoBody)
                        .foregroundStyle(Color.exoLightGray)
                        .listRowBackground(Color.exoDarkGray)
                }
            }
        } header: {
            sectionHeader("Status")
        }
    }
}
