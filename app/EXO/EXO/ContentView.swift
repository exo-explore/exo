//
//  ContentView.swift
//  EXO
//
//  Created by Sami Khan on 2025-11-22.
//

import AppKit
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var controller: ExoProcessController
    @EnvironmentObject private var stateService: ClusterStateService
    @EnvironmentObject private var networkStatusService: NetworkStatusService
    @EnvironmentObject private var localNetworkChecker: LocalNetworkChecker
    @EnvironmentObject private var updater: SparkleUpdater
    @State private var focusedNode: NodeViewModel?
    @State private var deletingInstanceIDs: Set<String> = []
    @State private var showAllNodes = false
    @State private var showAllInstances = false
    @State private var showDebugInfo = false
    @State private var bugReportInFlight = false
    @State private var bugReportMessage: String?
    @State private var showAdvancedOptions = false
    @State private var pendingNamespace: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            statusSection
            if shouldShowLocalNetworkWarning {
                localNetworkWarningBanner
            }
            if shouldShowClusterDetails {
                Divider()
                overviewSection
                topologySection
                nodeSection
            }
            if shouldShowInstances {
                instanceSection
            }
            Spacer(minLength: 0)
            controlButtons
        }
        .animation(.easeInOut(duration: 0.3), value: shouldShowClusterDetails)
        .animation(.easeInOut(duration: 0.3), value: shouldShowInstances)
        .animation(.easeInOut(duration: 0.3), value: shouldShowLocalNetworkWarning)
        .padding()
        .frame(width: 340)
        .onAppear {
            Task {
                await networkStatusService.refresh()
            }
        }
    }

    private var shouldShowLocalNetworkWarning: Bool {
        if case .notWorking = localNetworkChecker.status {
            return controller.status != .stopped
        }
        return false
    }

    private var localNetworkWarningBanner: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                Text("Local Network Access Issue")
                    .font(.caption)
                    .fontWeight(.semibold)
            }
            Text(
                "Device discovery won't work. To fix:\n1. Quit EXO\n2. Open System Settings → Privacy & Security → Local Network\n3. Toggle EXO off, then back on\n4. Relaunch EXO"
            )
            .font(.caption2)
            .foregroundColor(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            Button {
                openLocalNetworkSettings()
            } label: {
                Text("Open Settings")
                    .font(.caption2)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.orange.opacity(0.1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.orange.opacity(0.3), lineWidth: 1)
        )
    }

    private func openLocalNetworkSettings() {
        // Open Privacy & Security settings - Local Network section
        if let url = URL(
            string: "x-apple.systempreferences:com.apple.preference.security?Privacy_LocalNetwork")
        {
            NSWorkspace.shared.open(url)
        }
    }

    private var topologySection: some View {
        Group {
            if let topology = stateService.latestSnapshot?.topologyViewModel(
                localNodeId: stateService.localNodeId), !topology.nodes.isEmpty
            {
                TopologyMiniView(topology: topology)
            }
        }
    }

    private var statusSection: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text("EXO")
                    .font(.headline)
                Text(controller.status.displayText)
                    .font(.caption)
                    .foregroundColor(.secondary)
                if let detail = statusDetailText {
                    Text(detail)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            Spacer()
            Toggle("", isOn: processToggleBinding)
                .toggleStyle(.switch)
                .labelsHidden()
        }
    }

    private var overviewSection: some View {
        Group {
            if let snapshot = stateService.latestSnapshot {
                let overview = snapshot.overview()
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        VStack(alignment: .leading) {
                            Text(
                                "\(overview.usedRam, specifier: "%.0f") / \(overview.totalRam, specifier: "%.0f") GB"
                            )
                            .font(.headline)
                            Text("Memory")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        VStack(alignment: .leading) {
                            Text("\(overview.nodeCount)")
                                .font(.headline)
                            Text("Nodes")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        VStack(alignment: .leading) {
                            Text("\(overview.instanceCount)")
                                .font(.headline)
                            Text("Instances")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            } else {
                Text("Connecting to EXO…")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    private var nodeSection: some View {
        Group {
            if let nodes = stateService.latestSnapshot?.nodeViewModels(), !nodes.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Nodes")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("(\(nodes.count))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        collapseButton(isExpanded: $showAllNodes)
                    }
                    .animation(nil, value: showAllNodes)
                    if showAllNodes {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(nodes) { node in
                                NodeRowView(node: node)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 4)
                                    .background(.regularMaterial.opacity(0.6))
                                    .clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                        }
                        .transition(.opacity)
                    }
                }
                .animation(.easeInOut(duration: 0.25), value: showAllNodes)
            }
        }
    }

    private var instanceSection: some View {
        Group {
            if let instances = stateService.latestSnapshot?.instanceViewModels() {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Instances")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("(\(instances.count))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        if !instances.isEmpty {
                            collapseButton(isExpanded: $showAllInstances)
                        }
                    }
                    .animation(nil, value: showAllInstances)
                    if showAllInstances, !instances.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(instances) { instance in
                                InstanceRowView(instance: instance)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 4)
                                    .background(.regularMaterial.opacity(0.6))
                                    .clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                        }
                        .transition(.opacity)
                    }
                }
                .animation(.easeInOut(duration: 0.25), value: showAllInstances)
            }
        }
    }

    private var controlButtons: some View {
        VStack(alignment: .leading, spacing: 0) {
            if controller.status != .stopped {
                dashboardButton
                Divider()
                    .padding(.vertical, 8)
            } else {
                Divider()
                    .padding(.vertical, 4)
            }
            controlButton(title: "Check for Updates") {
                updater.checkForUpdates()
            }
            .padding(.bottom, 8)
            advancedOptionsSection
                .padding(.bottom, 8)
            debugSection
                .padding(.bottom, 8)
            controlButton(title: "Quit", tint: .secondary) {
                controller.stop()
                NSApplication.shared.terminate(nil)
            }
        }
    }

    private func controlButton(title: String, tint: Color = .primary, action: @escaping () -> Void)
        -> some View
    {
        HoverButton(title: title, tint: tint, trailingSystemImage: nil, action: action)
    }

    private var dashboardButton: some View {
        Button {
            guard let url = URL(string: "http://localhost:52415/") else { return }
            NSWorkspace.shared.open(url)
        } label: {
            HStack {
                Image(systemName: "arrow.up.right.square")
                    .imageScale(.small)
                Text("Dashboard")
                    .fontWeight(.medium)
                Spacer()
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 10)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color(red: 1.0, green: 0.87, blue: 0.0).opacity(0.2))
            )
        }
        .buttonStyle(.plain)
        .padding(.bottom, 4)
    }

    private func collapseButton(isExpanded: Binding<Bool>) -> some View {
        Button {
            isExpanded.wrappedValue.toggle()
        } label: {
            Label(
                isExpanded.wrappedValue ? "Hide" : "Show All",
                systemImage: isExpanded.wrappedValue ? "chevron.up" : "chevron.down"
            )
            .labelStyle(.titleAndIcon)
            .contentTransition(.symbolEffect(.replace))
        }
        .buttonStyle(.plain)
        .font(.caption2)
    }
    private func instancesToDisplay(_ instances: [InstanceViewModel]) -> [InstanceViewModel] {
        if showAllInstances {
            return instances
        }
        return []
    }

    private var shouldShowClusterDetails: Bool {
        controller.status != .stopped
    }

    private var shouldShowInstances: Bool {
        controller.status != .stopped
    }

    private var statusDetailText: String? {
        switch controller.status {
        case .failed(let message):
            return message
        case .stopped:
            if let countdown = controller.launchCountdownSeconds {
                return "Launching in \(countdown)s"
            }
            return nil
        default:
            if let countdown = controller.launchCountdownSeconds {
                return "Launching in \(countdown)s"
            }
            if let lastError = controller.lastError {
                return lastError
            }
            if let message = stateService.lastActionMessage {
                return message
            }
            return nil
        }
    }

    private var thunderboltStatusText: String {
        switch networkStatusService.status.thunderboltBridgeState {
        case .some(.disabled):
            return "Thunderbolt Bridge: Disabled"
        case .some(.deleted):
            return "Thunderbolt Bridge: Deleted"
        case .some(.enabled):
            return "Thunderbolt Bridge: Enabled"
        case nil:
            return "Thunderbolt Bridge: Unknown"
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

    private var advancedOptionsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Advanced Options")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                collapseButton(isExpanded: $showAdvancedOptions)
            }
            .animation(nil, value: showAdvancedOptions)
            if showAdvancedOptions {
                VStack(alignment: .leading, spacing: 8) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Cluster Namespace")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        HStack {
                            TextField("optional", text: $pendingNamespace)
                                .textFieldStyle(.roundedBorder)
                                .font(.caption2)
                                .onAppear {
                                    pendingNamespace = controller.customNamespace
                                }
                            Button("Save & Restart") {
                                controller.customNamespace = pendingNamespace
                                if controller.status == .running || controller.status == .starting {
                                    controller.restart()
                                }
                            }
                            .font(.caption2)
                            .disabled(pendingNamespace == controller.customNamespace)
                        }

                    }
                }
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: showAdvancedOptions)
    }

    private var debugSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Debug Info")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                collapseButton(isExpanded: $showDebugInfo)
            }
            .animation(nil, value: showDebugInfo)
            if showDebugInfo {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Version: \(buildTag)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text("Commit: \(buildCommit)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text(thunderboltStatusText)
                        .font(.caption2)
                        .foregroundColor(thunderboltStatusColor)
                    interfaceIpList
                    rdmaStatusView
                    sendBugReportButton
                        .padding(.top, 6)
                }
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: showDebugInfo)
    }

    private var rdmaStatusView: some View {
        let rdma = networkStatusService.status.rdmaStatus
        return VStack(alignment: .leading, spacing: 1) {
            Text("RDMA: \(rdmaStatusText(rdma))")
                .font(.caption2)
                .foregroundColor(rdmaStatusColor(rdma))
            if !rdma.devices.isEmpty {
                Text("  Devices: \(rdma.devices.joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            if !rdma.activePorts.isEmpty {
                Text("  Active Ports:")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                ForEach(rdma.activePorts, id: \.device) { port in
                    Text("    \(port.device) port \(port.port): \(port.state)")
                        .font(.caption2)
                        .foregroundColor(.green)
                }
            }
        }
    }

    private func rdmaStatusText(_ rdma: RDMAStatus) -> String {
        switch rdma.rdmaCtlEnabled {
        case .some(true):
            return "Enabled"
        case .some(false):
            return "Disabled"
        case nil:
            return rdma.devices.isEmpty ? "Not Available" : "Available"
        }
    }

    private func rdmaStatusColor(_ rdma: RDMAStatus) -> Color {
        switch rdma.rdmaCtlEnabled {
        case .some(true):
            return .green
        case .some(false):
            return .orange
        case nil:
            return rdma.devices.isEmpty ? .secondary : .green
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
                .padding(.vertical, 6)
                .padding(.horizontal, 8)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.accentColor.opacity(0.12))
                )
            }
            .buttonStyle(.plain)
            .disabled(bugReportInFlight)

            if let message = bugReportMessage {
                Text(message)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var processToggleBinding: Binding<Bool> {
        Binding(
            get: {
                switch controller.status {
                case .running, .starting:
                    return true
                case .stopped, .failed:
                    return false
                }
            },
            set: { isOn in
                if isOn {
                    stateService.resetTransientState()
                    stateService.startPolling()
                    controller.cancelPendingLaunch()
                    controller.launchIfNeeded()
                } else {
                    stateService.stopPolling()
                    controller.stop()
                    stateService.resetTransientState()
                }
            }
        )
    }

    private func bindingForNode(_ node: NodeViewModel) -> Binding<NodeViewModel?> {
        Binding<NodeViewModel?>(
            get: {
                focusedNode?.id == node.id ? focusedNode : nil
            },
            set: { newValue in
                if newValue == nil {
                    focusedNode = nil
                } else {
                    focusedNode = newValue
                }
            }
        )
    }

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

    private var buildTag: String {
        Bundle.main.infoDictionary?["EXOBuildTag"] as? String ?? "unknown"
    }

    private var buildCommit: String {
        Bundle.main.infoDictionary?["EXOBuildCommit"] as? String ?? "unknown"
    }
}

private struct HoverButton: View {
    let title: String
    let tint: Color
    let trailingSystemImage: String?
    let action: () -> Void

    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack {
                Text(title)
                Spacer()
                if let systemName = trailingSystemImage {
                    Image(systemName: systemName)
                        .imageScale(.small)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 6)
            .padding(.horizontal, 8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(
                        isHovering
                            ? Color.accentColor.opacity(0.1)
                            : Color.clear
                    )
            )
        }
        .buttonStyle(.plain)
        .foregroundColor(tint)
        .onHover { isHovering = $0 }
    }
}
