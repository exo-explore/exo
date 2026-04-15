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
    @State private var pendingHFEndpoint: String = ""
    @State private var pendingEnableImageModels = false
    @State private var pendingOfflineMode = false
    @State private var pendingFastSynchEnabled = false
    @State private var pendingDefaultModelsDir: String = ""
    @State private var pendingAdditionalModelsDirs: String = ""
    @State private var pendingReadOnlyModelsDirs: String = ""
    @State private var pendingCustomEnvironmentVariables: [CustomEnvironmentVariable] = []
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
            environmentTab
                .tabItem {
                    Label("Environment", systemImage: "terminal")
                }
            aboutTab
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
        }
        .frame(width: 640, height: 560)
        .onAppear {
            pendingNamespace = controller.customNamespace
            pendingHFToken = controller.hfToken
            pendingHFEndpoint = controller.hfEndpoint
            pendingEnableImageModels = controller.enableImageModels
            pendingOfflineMode = controller.offlineMode
            pendingFastSynchEnabled = controller.fastSynchEnabled
            pendingDefaultModelsDir = controller.defaultModelsDir
            pendingAdditionalModelsDirs = controller.additionalModelsDirs
            pendingReadOnlyModelsDirs = controller.readOnlyModelsDirs
            pendingCustomEnvironmentVariables = controller.customEnvironmentVariables
            needsRestart = false
        }
    }

    // MARK: - General Tab

    private var generalTab: some View {
        Form {
            Section {
                LabeledContent("Cluster Namespace") {
                    TextField("", text: $pendingNamespace, prompt: Text("default"))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 260)
                }
                Text("Nodes with the same namespace form a cluster. Leave empty for default.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section {
                LabeledContent("HuggingFace Token") {
                    SecureField("", text: $pendingHFToken, prompt: Text("optional"))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 260)
                }
                Text("Required for gated models. Get yours at huggingface.co/settings/tokens")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section {
                LabeledContent("HuggingFace Endpoint") {
                    TextField("", text: $pendingHFEndpoint, prompt: Text("default"))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 260)
                }
                Text("Defaults to huggingface.co. Use a mirror (e.g. hf-mirror.com) for China.")
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
            Section("Performance") {
                Toggle("Fast Synch Enabled", isOn: $pendingFastSynchEnabled)
                Text(
                    "Experimental: enables fast CPU to GPU synchronization. Can sometimes cause a \"GPU lock\" where inference hangs for ~10 seconds before starting. Necessary for low latency with RDMA and Tensor Parallelism."
                )
                .font(.caption)
                .foregroundColor(.secondary)

                HStack {
                    Spacer()
                    Button("Save & Restart") {
                        applyAdvancedSettings()
                    }
                    .disabled(!hasAdvancedChanges)
                }
            }

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

    // MARK: - Environment Tab

    private var environmentTab: some View {
        Form {
            Section("Models Directories") {
                LabeledContent("Default Models Directory") {
                    TextField(
                        "",
                        text: $pendingDefaultModelsDir,
                        prompt: Text("~/.exo/models")
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 260)
                }
                Text("Sets EXO_DEFAULT_MODELS_DIR. Where models are downloaded.")
                    .font(.caption)
                    .foregroundColor(.secondary)

                LabeledContent("Additional Directories") {
                    TextField(
                        "",
                        text: $pendingAdditionalModelsDirs,
                        prompt: Text("optional, colon-separated")
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 260)
                }
                Text("Sets EXO_MODELS_DIRS. Extra writable model directories.")
                    .font(.caption)
                    .foregroundColor(.secondary)

                LabeledContent("Read-Only Directories") {
                    TextField(
                        "",
                        text: $pendingReadOnlyModelsDirs,
                        prompt: Text("optional, colon-separated")
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 260)
                }
                Text("Sets EXO_MODELS_READ_ONLY_DIRS. Never written to.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section("Custom Environment Variables") {
                Text(
                    "Escape hatch for env vars that don't have typed fields above. "
                        + "Values here override the typed fields on conflict."
                )
                .font(.caption)
                .foregroundColor(.secondary)

                if pendingCustomEnvironmentVariables.isEmpty {
                    Text("No custom variables.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    ForEach($pendingCustomEnvironmentVariables) { $variable in
                        HStack(alignment: .center, spacing: 8) {
                            VStack(spacing: 4) {
                                TextField("key", text: $variable.key)
                                    .labelsHidden()
                                    .textFieldStyle(.roundedBorder)
                                    .font(.system(.body, design: .monospaced))
                                TextField("value", text: $variable.value)
                                    .labelsHidden()
                                    .textFieldStyle(.roundedBorder)
                                    .font(.system(.body, design: .monospaced))
                            }
                            VStack(spacing: 4) {
                                Button {
                                    pendingCustomEnvironmentVariables.removeAll {
                                        $0.id == variable.id
                                    }
                                } label: {
                                    Image(systemName: "minus.circle")
                                }
                                .buttonStyle(.borderless)
                                .help("Remove variable")
                                if !isValidEnvironmentVariableName(variable.key) {
                                    Image(systemName: "exclamationmark.triangle.fill")
                                        .foregroundColor(.orange)
                                        .help(
                                            "Invalid environment variable name. "
                                                + "Must match [A-Za-z_][A-Za-z0-9_]*."
                                        )
                                }
                            }
                        }
                    }
                }

                HStack {
                    Button {
                        pendingCustomEnvironmentVariables.append(
                            CustomEnvironmentVariable()
                        )
                    } label: {
                        Label("Add Variable", systemImage: "plus")
                    }
                    Spacer()
                }
            }

            Section {
                HStack {
                    Spacer()
                    Button("Save & Restart") {
                        applyEnvironmentSettings()
                    }
                    .disabled(!hasEnvironmentChanges)
                }
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
            || pendingHFEndpoint != controller.hfEndpoint
            || pendingOfflineMode != controller.offlineMode
    }

    private var hasModelChanges: Bool {
        pendingEnableImageModels != controller.enableImageModels
    }

    private var hasAdvancedChanges: Bool {
        pendingFastSynchEnabled != controller.fastSynchEnabled
    }

    private var hasEnvironmentChanges: Bool {
        pendingDefaultModelsDir != controller.defaultModelsDir
            || pendingAdditionalModelsDirs != controller.additionalModelsDirs
            || pendingReadOnlyModelsDirs != controller.readOnlyModelsDirs
            || pendingCustomEnvironmentVariables != controller.customEnvironmentVariables
    }

    private func applyGeneralSettings() {
        controller.customNamespace = pendingNamespace
        controller.hfToken = pendingHFToken
        controller.hfEndpoint = pendingHFEndpoint
        controller.offlineMode = pendingOfflineMode
        restartIfRunning()
    }

    private func applyModelSettings() {
        controller.enableImageModels = pendingEnableImageModels
        restartIfRunning()
    }

    private func applyAdvancedSettings() {
        controller.fastSynchEnabled = pendingFastSynchEnabled
        restartIfRunning()
    }

    private func applyEnvironmentSettings() {
        controller.defaultModelsDir = pendingDefaultModelsDir.trimmingCharacters(
            in: .whitespaces)
        controller.additionalModelsDirs = pendingAdditionalModelsDirs.trimmingCharacters(
            in: .whitespaces)
        controller.readOnlyModelsDirs = pendingReadOnlyModelsDirs.trimmingCharacters(
            in: .whitespaces)

        pendingDefaultModelsDir = controller.defaultModelsDir
        pendingAdditionalModelsDirs = controller.additionalModelsDirs
        pendingReadOnlyModelsDirs = controller.readOnlyModelsDirs

        // Trim whitespace from keys and drop empty ones so that the stored
        // form matches what is actually injected into the child process and
        // hasEnvironmentChanges doesn't show a stale diff after save.
        let trimmed: [CustomEnvironmentVariable] =
            pendingCustomEnvironmentVariables.compactMap { variable in
                let key = variable.key.trimmingCharacters(in: .whitespaces)
                guard !key.isEmpty else { return nil }
                return CustomEnvironmentVariable(
                    id: variable.id, key: key, value: variable.value
                )
            }

        // De-duplicate keys, keeping the last occurrence. This matches the
        // effective semantics of the dictionary assignment in
        // ExoProcessController.makeEnvironment and avoids silently losing
        // visible rows after save.
        var seenKeys = Set<String>()
        var deduplicatedReversed: [CustomEnvironmentVariable] = []
        for variable in trimmed.reversed() {
            if seenKeys.insert(variable.key).inserted {
                deduplicatedReversed.append(variable)
            }
        }
        let sanitized = Array(deduplicatedReversed.reversed())

        pendingCustomEnvironmentVariables = sanitized
        controller.customEnvironmentVariables = sanitized

        restartIfRunning()
    }

    /// Validates a POSIX-style environment variable name:
    /// `[A-Za-z_][A-Za-z0-9_]*`. Uses an ASCII-only charset so that
    /// Unicode letters (e.g. `ñ`, Cyrillic) are rejected in line with what
    /// the help tooltip advertises. Empty strings are treated as valid
    /// here so that a freshly added blank row does not immediately look
    /// broken; the save step filters empty keys out instead.
    private func isValidEnvironmentVariableName(_ key: String) -> Bool {
        if key.isEmpty { return true }
        let headAllowed = CharacterSet(
            charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
        )
        let tailAllowed = headAllowed.union(CharacterSet(charactersIn: "0123456789"))
        guard let first = key.unicodeScalars.first, headAllowed.contains(first) else {
            return false
        }
        for scalar in key.unicodeScalars.dropFirst() {
            if !tailAllowed.contains(scalar) { return false }
        }
        return true
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
