import AppKit
import ServiceManagement
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
    @State private var launchAtLoginStatus: SMAppService.Status = .notRegistered
    @State private var launchAtLoginError: String?
    @AppStorage(StartupPreferences.openDashboardOnStartupKey)
    private var openDashboardOnStartup = StartupPreferences.openDashboardOnStartupDefault
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
        .onReceive(
            NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)
        ) { _ in
            // Pick up changes made in System Settings → Login Items.
            launchAtLoginStatus = LaunchAtLoginHelper.status
        }
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
            launchAtLoginStatus = LaunchAtLoginHelper.status
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

            Section("Startup") {
                Text("These apply immediately, without Save & Restart.")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Toggle("Launch at login", isOn: launchAtLoginBinding)
                Text("Start EXO automatically when you log in to this Mac.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                if let launchAtLoginError {
                    Text(launchAtLoginError)
                        .font(.caption)
                        .foregroundColor(.orange)
                } else if launchAtLoginStatus == .requiresApproval {
                    HStack {
                        Text("Allow EXO in System Settings → General → Login Items.")
                            .font(.caption)
                            .foregroundColor(.orange)
                        Spacer()
                        Button("Open Login Items") {
                            SMAppService.openSystemSettingsLoginItems()
                        }
                    }
                }

                Toggle("Open dashboard on startup", isOn: $openDashboardOnStartup)
                Text("Open the web dashboard in your browser when EXO starts.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    /// Reflects the login item's actual state in System Settings rather than a
    /// stored copy, so it stays right if the user changes it there.
    private var launchAtLoginBinding: Binding<Bool> {
        Binding(
            get: {
                launchAtLoginStatus == .enabled || launchAtLoginStatus == .requiresApproval
            },
            set: { enabled in
                // The user has decided, so startup no longer changes it.
                StartupPreferences().markLaunchAtLoginDefaultApplied()
                do {
                    try LaunchAtLoginHelper.setEnabled(enabled)
                    launchAtLoginError = nil
                } catch {
                    launchAtLoginError =
                        "Couldn't \(enabled ? "turn on" : "turn off") launch at login: "
                        + error.localizedDescription
                }
                launchAtLoginStatus = LaunchAtLoginHelper.status
            }
        )
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
                    // Rows bind to their variable by id rather than by array
                    // index (as `ForEach($pendingCustomEnvironmentVariables)`
                    // would), so nothing can read or write through a stale
                    // index after a row is removed.
                    ForEach(pendingCustomEnvironmentVariables) { variable in
                        HStack(alignment: .center, spacing: 8) {
                            VStack(spacing: 4) {
                                TextField(
                                    "key",
                                    text: $pendingCustomEnvironmentVariables.field(
                                        \.key, of: variable.id)
                                )
                                .labelsHidden()
                                .textFieldStyle(.roundedBorder)
                                .font(.system(.body, design: .monospaced))
                                TextField(
                                    "value",
                                    text: $pendingCustomEnvironmentVariables.field(
                                        \.value, of: variable.id)
                                )
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
                                if variable.hasInvalidName {
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

                if let invalidNamesHint {
                    Text(invalidNamesHint)
                        .font(.caption)
                        .foregroundColor(.orange)
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
                    .disabled(
                        !hasEnvironmentChanges || !invalidCustomEnvironmentVariableNames.isEmpty)
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

    // MARK: - Actions

    private func showUninstallConfirmationAlert() {
        let alert = NSAlert()
        alert.messageText = "Uninstall EXO"
        alert.informativeText = """
            This will remove EXO and all its components:

            • Network configuration daemon
            • Launch at login registration
            • EXO network location
            • EXO data directory (~/.exo)

            The app will be moved to Trash.
            """
        alert.alertStyle = .warning

        let checkbox = NSButton(
            checkboxWithTitle: "Keep downloaded models (~/.exo/models)",
            target: nil, action: nil)
        checkbox.state = .off
        checkbox.sizeToFit()
        alert.accessoryView = checkbox

        alert.addButton(withTitle: "Uninstall")
        alert.addButton(withTitle: "Cancel")

        if let uninstallButton = alert.buttons.first {
            uninstallButton.hasDestructiveAction = true
        }

        let response = alert.runModal()
        if response == .alertFirstButtonReturn {
            performUninstall(keepModels: checkbox.state == .on)
        }
    }

    private func performUninstall(keepModels: Bool) {
        uninstallInProgress = true

        controller.cancelPendingLaunch()
        controller.stop()
        stateService.stopPolling()

        DispatchQueue.global(qos: .utility).async {
            do {
                try NetworkSetupHelper.uninstall()
                try Self.removeExoDirectory(keepModels: keepModels)

                DispatchQueue.main.async {
                    LaunchAtLoginHelper.disable()
                    StartupPreferences().resetForUninstall()
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

    private static func removeExoDirectory(keepModels: Bool) throws {
        let fm = FileManager.default
        let exoDir = ExoProcessController.exoDirectoryURL
        guard fm.fileExists(atPath: exoDir.path) else { return }

        if !keepModels {
            try fm.removeItem(at: exoDir)
            return
        }

        let contents = try fm.contentsOfDirectory(
            at: exoDir, includingPropertiesForKeys: nil, options: [])
        for entry in contents where entry.lastPathComponent != "models" {
            try? fm.removeItem(at: entry)
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

    private var invalidCustomEnvironmentVariableNames: [String] {
        pendingCustomEnvironmentVariables.filter(\.hasInvalidName).map(\.trimmedKey)
    }

    /// Names the rows that block saving, since a variable saved by an older
    /// version can block saving the unrelated fields in this tab too.
    private var invalidNamesHint: String? {
        let names = invalidCustomEnvironmentVariableNames
        guard !names.isEmpty else { return nil }
        let list = names.map { "\"\($0)\"" }.joined(separator: ", ")
        let one = names.count == 1
        return "Invalid variable name\(one ? "" : "s"): \(list). "
            + "Fix or remove \(one ? "it" : "them") to save this tab. "
            + "Names must match [A-Za-z_][A-Za-z0-9_]*."
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

        // Store exactly what is injected into the child process (trimmed
        // keys, no blank or invalid rows, last duplicate wins) so that
        // hasEnvironmentChanges doesn't show a stale diff after save.
        let sanitized = CustomEnvironmentVariable.sanitized(pendingCustomEnvironmentVariables)

        pendingCustomEnvironmentVariables = sanitized
        controller.customEnvironmentVariables = sanitized

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
