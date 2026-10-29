import AppKit
import SwiftUI

/// Settings › Cluster: which devices this Mac joins, and its connections.
struct ClusterSettingsPane: View {
    @EnvironmentObject private var controller: ExoProcessController
    @EnvironmentObject private var localNetworkChecker: LocalNetworkChecker
    @EnvironmentObject private var networkStatusService: NetworkStatusService

    @State private var thunderboltNetworking = NetworkSetupHelper.setupState
    @State private var thunderboltNetworkingBusy = false

    var body: some View {
        Form {
            Section {
                SettingTextField(
                    title: "Cluster name",
                    value: $controller.customNamespace,
                    prompt: Text("Automatic (EXO \(ExoProcessController.buildTag))"),
                    normalize: { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                )
                .help("Passed to exo as EXO_ZENOH_NAMESPACE.")
            } footer: {
                SettingFootnote(
                    "Devices on your network with the same cluster name work together. "
                        + "Automatic joins only devices running the same version of EXO.")
            }

            Section {
                LabeledContent("Local network access") {
                    localNetworkStatus
                }
                LabeledContent("Thunderbolt networking") {
                    thunderboltNetworkingStatus
                }
                LabeledContent("Thunderbolt Bridge") {
                    thunderboltBridgeStatus
                }
            } header: {
                Text("Connections")
            } footer: {
                SettingFootnote(
                    "EXO's network service keeps Thunderbolt Bridge off, so Macs connected "
                        + "by Thunderbolt in a loop don't flood the network. Setting it up or "
                        + "removing it needs an administrator password.")
            }
        }
        .settingsPane()
        .task {
            // Only while this pane is visible.
            while !Task.isCancelled {
                thunderboltNetworking = NetworkSetupHelper.setupState
                await networkStatusService.refresh()
                try? await Task.sleep(for: .seconds(3))
            }
        }
    }

    // MARK: Local network

    @ViewBuilder private var localNetworkStatus: some View {
        switch localNetworkChecker.status {
        case .working:
            Label("Allowed", systemImage: "checkmark.circle")
        case .notWorking:
            HStack {
                Label("Off", systemImage: "exclamationmark.triangle")
                Button("Open Privacy & Security…") {
                    guard
                        let url = URL(
                            string:
                                "x-apple.systempreferences:com.apple.preference.security?Privacy_LocalNetwork"
                        )
                    else { return }
                    NSWorkspace.shared.open(url)
                }
            }
            .help("EXO can't find other devices without local network access.")
        case .unknown, .checking:
            Text("Checking…")
                .foregroundStyle(.secondary)
        }
    }

    // MARK: Thunderbolt networking

    @ViewBuilder private var thunderboltNetworkingStatus: some View {
        HStack {
            switch thunderboltNetworking {
            case .current:
                Label("Set up", systemImage: "checkmark.circle")
                Button("Remove…", action: removeThunderboltNetworking)
            case .outdated:
                Label("Needs an update", systemImage: "exclamationmark.triangle")
                Button("Update…", action: setUpThunderboltNetworking)
            case .notInstalled:
                Label("Not set up", systemImage: "minus.circle")
                Button("Set Up…", action: setUpThunderboltNetworking)
            }
        }
        .disabled(thunderboltNetworkingBusy)
    }

    /// Explains what changes, then asks for the password. Offered here even
    /// after "Not Now" in the menu.
    private func setUpThunderboltNetworking() {
        thunderboltNetworkingBusy = true
        Task {
            await NetworkSetupHelper.offerSetUp()
            thunderboltNetworkingBusy = false
            thunderboltNetworking = NetworkSetupHelper.setupState
            SettingsWindowController.bringSettingsToFront()
        }
    }

    private func removeThunderboltNetworking() {
        let alert = NSAlert()
        alert.messageText = "Remove Thunderbolt Networking?"
        alert.informativeText =
            "EXO's network service will be removed and Thunderbolt Bridge turned back on. "
            + "You'll be asked for an administrator password."
        alert.addButton(withTitle: "Remove")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModalInFront() == .alertFirstButtonReturn else { return }

        thunderboltNetworkingBusy = true
        Task {
            let result = await Task.detached(priority: .utility) {
                Result { try NetworkSetupHelper.uninstall() }
            }.value
            thunderboltNetworkingBusy = false
            thunderboltNetworking = NetworkSetupHelper.setupState
            if case .failure(let error) = result, error as? NetworkSetupError != .cancelled {
                let failure = NSAlert()
                failure.messageText = "Thunderbolt Networking Wasn't Removed"
                failure.informativeText = error.localizedDescription
                failure.alertStyle = .warning
                failure.runModalInFront()
            }
            SettingsWindowController.bringSettingsToFront()
        }
    }

    // MARK: Thunderbolt Bridge

    @ViewBuilder private var thunderboltBridgeStatus: some View {
        switch networkStatusService.status.thunderboltBridgeState {
        case .disabled, .deleted:
            Label("Off (recommended)", systemImage: "checkmark.circle")
        case .enabled:
            Label("On", systemImage: "exclamationmark.triangle")
                .help(
                    "Thunderbolt Bridge can flood the network when Macs are connected "
                        + "by Thunderbolt in a loop.")
        case nil:
            Text("None")
                .foregroundStyle(.secondary)
        }
    }
}
