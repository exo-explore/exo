import AppKit
import SwiftUI

/// Native macOS Settings window following Apple HIG.
/// Organized into General, Model, Advanced, and About sections.
struct SettingsView: View {
    @EnvironmentObject private var controller: ExoProcessController
    @EnvironmentObject private var updater: SparkleUpdater
    @EnvironmentObject private var networkStatusService: NetworkStatusService
    @EnvironmentObject private var thunderboltBridgeService: ThunderboltBridgeService
    @EnvironmentObject private var stateService: ClusterStateService

    @State private var pendingNamespace: String = ""
    @State private var pendingHFToken: String = ""
    @State private var pendingEnableImageModels = false
    @State private var pendingOfflineMode = false
    @State private var needsRestart = false
    @State private var bugReportInFlight = false
    @State private var bugReportMessage: String?
    @State private var uninstallInProgress = false

    var body: some View {
        TabView {
            generalTab
                .tabItem {
                    Label("General", systemImage: "gear")
                }
            modelTab
                .tabItem {
                    Label("Model", systemImage: "cube")
                }
            advancedTab
                .tabItem {
                    Label("Advanced", systemImage: "wrench.and.screwdriver")
                }
            aboutTab
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
        }
        .frame(width: 450, height: 400)
        .onAppear {
            pendingNamespace = controller.customNamespace
            pendingHFToken = controller.hfToken
            pendingEnableImageModels = controller.enableImageModels
            pendingOfflineMode = controller.offlineMode
            needsRestart = false
        }
    }

    // MARK: - General Tab

    private var generalTab: some View {
        Form {
            Section {
                LabeledContent("Cluster Namespace") {
                    TextField("default", text: $pendingNamespace)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 200)
                }
                Text("Nodes with the same namespace form a cluster. Leave empty for default.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section {
                LabeledContent("HuggingFace Token") {
                    SecureField("optional", text: $pendingHFToken)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 200)
                }
                Text("Required for gated models. Get yours at huggingface.co/settings/tokens")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section {
                Toggle("Offline Mode", isOn: $pendingOfflineMode)
                Text("Skip internet checks and use only locally available models.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section {
                HStack {
                    Spacer()
                    Button("Save & Restart") {
                        applyGeneralSettings()
                    }
                    .disabled(!hasGeneralChanges)
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - Model Tab

    private var modelTab: some View {
        Form {
            Section {
                Toggle("Enable Image Models (experimental)", isOn: $pendingEnableImageModels)
                Text("Allow text-to-image and image-to-image models in the model picker.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section {
                HStack {
                    Spacer()
                    Button("Save & Restart") {
                        applyModelSettings()
                    }
                    .disabled(!hasModelChanges)
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - Advanced Tab

    private var advancedTab: some View {
        Form {
            Section("Onboarding") {
                HStack {
                    VStack(alignment: .leading) {
                        Text("Reset Onboarding")
                        Text("Opens the dashboard and resets the onboarding wizard.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Button("Reset") {
                        guard let url = URL(string: "http://localhost:52415/?reset-onboarding")
                        else { return }
                        NSWorkspace.shared.open(url)
                    }
                }
            }

            Section("Debug Info") {
                LabeledContent("Thunderbolt Bridge") {
                    Text(thunderboltStatusText)
                        .foregroundColor(thunderboltStatusColor)
                }

                VStack(alignment: .leading, spacing: 2) {
                    clusterThunderboltBridgeView
                }

                VStack(alignment: .leading, spacing: 2) {
                    interfaceIpList
                }

                VStack(alignment: .leading, spacing: 2) {
                    rdmaStatusView
                }

                sendBugReportButton
            }

            Section("Danger Zone") {
                Button(role: .destructive) {
                    showUninstallConfirmationAlert()
                } label: {
                    HStack {
                        Text("Uninstall EXO")
                        Spacer()
                        Image(systemName: "trash")
                            .imageScale(.small)
                    }
                }
                .disabled(uninstallInProgress)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - About Tab

    private var aboutTab: some View {
        Form {
            Section {
                LabeledContent("Version") {
                    Text(buildTag)
                        .textSelection(.enabled)
                }
                LabeledContent("Commit") {
                    Text(buildCommit)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                }
            }

            Section {
                Button("Check for Updates") {
                    updater.checkForUpdates()
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - Debug Info Views (moved from ContentView)

    private var thunderboltStatusText: String {
        switch networkStatusService.status.thunderboltBridgeState {
        case .some(.disabled):
            return "Disabled"
        case .some(.deleted):
            return "Deleted"
        case .some(.enabled):
            return "Enabled"
        case nil:
            return "Unknown"
        }
    }

    private var thunderboltStatusColor: Color {
        switch networkStatusService.status.thunderboltBridgeState {
        case .some(.disabled), .some(.deleted):
            return .green
        case .some(.enabled):
            return .red
        case nil:
            return .secondary
        }
    }

    private var clusterThunderboltBridgeView: some View {
        let bridgeStatuses = stateService.latestSnapshot?.nodeThunderboltBridge ?? [:]
        let localNodeId = stateService.localNodeId
        let nodeProfiles = stateService.latestSnapshot?.nodeProfiles ?? [:]

        return VStack(alignment: .leading, spacing: 1) {
            if bridgeStatuses.isEmpty {
                Text("Cluster TB Bridge: No data")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            } else {
                Text("Cluster TB Bridge Status:")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                ForEach(Array(bridgeStatuses.keys.sorted()), id: \.self) { nodeId in
                    if let status = bridgeStatuses[nodeId] {
                        let nodeName =
                            nodeProfiles[nodeId]?.friendlyName ?? String(nodeId.prefix(8))
                        let isLocal = nodeId == localNodeId
                        let prefix = isLocal ? "  \(nodeName) (local):" : "  \(nodeName):"
                        let statusText =
                            !status.exists
                            ? "N/A"
                            : (status.enabled ? "Enabled" : "Disabled")
                        let color: Color =
                            !status.exists
                            ? .secondary
                            : (status.enabled ? .red : .green)
                        Text("\(prefix) \(statusText)")
                            .font(.caption2)
                            .foregroundColor(color)
                    }
                }
            }
        }
    }

    private var interfaceIpList: some View {
        let statuses = networkStatusService.status.interfaceStatuses
        return VStack(alignment: .leading, spacing: 1) {
            Text("Interfaces (en0–en7):")
                .font(.caption2)
                .foregroundColor(.secondary)
            if statuses.isEmpty {
                Text("  Unknown")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            } else {
                ForEach(statuses, id: \.interfaceName) { status in
                    let ipText = status.ipAddress ?? "No IP"
                    Text("  \(status.interfaceName): \(ipText)")
                        .font(.caption2)
                        .foregroundColor(status.ipAddress == nil ? .red : .green)
                }
            }
        }
    }

    private var rdmaStatusView: some View {
        let rdmaStatuses = stateService.latestSnapshot?.nodeRdmaCtl ?? [:]
        let localNodeId = stateService.localNodeId
        let nodeProfiles = stateService.latestSnapshot?.nodeProfiles ?? [:]
        let localDevices = networkStatusService.status.localRdmaDevices
        let localPorts = networkStatusService.status.localRdmaActivePorts

        return VStack(alignment: .leading, spacing: 1) {
            if rdmaStatuses.isEmpty {
                Text("Cluster RDMA: No data")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            } else {
                Text("Cluster RDMA Status:")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                ForEach(Array(rdmaStatuses.keys.sorted()), id: \.self) { nodeId in
                    if let status = rdmaStatuses[nodeId] {
                        let nodeName =
                            nodeProfiles[nodeId]?.friendlyName ?? String(nodeId.prefix(8))
                        let isLocal = nodeId == localNodeId
                        let prefix = isLocal ? "  \(nodeName) (local):" : "  \(nodeName):"
                        let statusText = status.enabled ? "Enabled" : "Disabled"
                        let color: Color = status.enabled ? .green : .orange
                        Text("\(prefix) \(statusText)")
                            .font(.caption2)
                            .foregroundColor(color)
                    }
                }
            }
            if !localDevices.isEmpty {
                Text("  Local Devices: \(localDevices.joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            if !localPorts.isEmpty {
                Text("  Local Active Ports:")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                ForEach(localPorts, id: \.device) { port in
                    Text("    \(port.device) port \(port.port): \(port.state)")
                        .font(.caption2)
                        .foregroundColor(.green)
                }
            }
        }
    }

    private var sendBugReportButton: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button {
                Task {
                    await sendBugReport()
                }
            } label: {
                HStack {
                    if bugReportInFlight {
                        ProgressView()
                            .scaleEffect(0.6)
                    }
                    Text("Send Bug Report")
                        .font(.caption)
                        .fontWeight(.semibold)
                    Spacer()
                }
            }
            .disabled(bugReportInFlight)

            if let message = bugReportMessage {
                Text(message)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Actions

    private func sendBugReport() async {
        bugReportInFlight = true
        bugReportMessage = "Collecting logs..."
        let service = BugReportService()
        do {
            let outcome = try await service.sendReport(isManual: true)
            bugReportMessage = outcome.message
        } catch {
            bugReportMessage = error.localizedDescription
        }
        bugReportInFlight = false
    }

    private func showUninstallConfirmationAlert() {
        let alert = NSAlert()
        alert.messageText = "Uninstall EXO"
        alert.informativeText = """
            This will remove EXO and all its system components:

            • Network configuration daemon
            • Launch at login registration
            • EXO network location

            The app will be moved to Trash.
            """
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Uninstall")
        alert.addButton(withTitle: "Cancel")

        if let uninstallButton = alert.buttons.first {
            uninstallButton.hasDestructiveAction = true
        }

        let response = alert.runModal()
        if response == .alertFirstButtonReturn {
            performUninstall()
        }
    }

    private func performUninstall() {
        uninstallInProgress = true

        controller.cancelPendingLaunch()
        controller.stop()
        stateService.stopPolling()

        DispatchQueue.global(qos: .utility).async {
            do {
                try NetworkSetupHelper.uninstall()

                DispatchQueue.main.async {
                    LaunchAtLoginHelper.disable()
                    self.moveAppToTrash()

                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        NSApplication.shared.terminate(nil)
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    let errorAlert = NSAlert()
                    errorAlert.messageText = "Uninstall Failed"
                    errorAlert.informativeText = error.localizedDescription
                    errorAlert.alertStyle = .critical
                    errorAlert.addButton(withTitle: "OK")
                    errorAlert.runModal()
                    self.uninstallInProgress = false
                }
            }
        }
    }

    private func moveAppToTrash() {
        guard let appURL = Bundle.main.bundleURL as URL? else { return }
        do {
            try FileManager.default.trashItem(at: appURL, resultingItemURL: nil)
        } catch {
            // If we can't trash the app, that's OK - user can do it manually
        }
    }

    // MARK: - Helpers

    private var hasGeneralChanges: Bool {
        pendingNamespace != controller.customNamespace || pendingHFToken != controller.hfToken
            || pendingOfflineMode != controller.offlineMode
    }

    private var hasModelChanges: Bool {
        pendingEnableImageModels != controller.enableImageModels
    }

    private func applyGeneralSettings() {
        controller.customNamespace = pendingNamespace
        controller.hfToken = pendingHFToken
        controller.offlineMode = pendingOfflineMode
        restartIfRunning()
    }

    private func applyModelSettings() {
        controller.enableImageModels = pendingEnableImageModels
        restartIfRunning()
    }

    private func restartIfRunning() {
        if controller.status == .running || controller.status == .starting {
            controller.restart()
        }
    }

    private var buildTag: String {
        Bundle.main.infoDictionary?["EXOBuildTag"] as? String ?? "unknown"
    }

    private var buildCommit: String {
        Bundle.main.infoDictionary?["EXOBuildCommit"] as? String ?? "unknown"
    }
}
