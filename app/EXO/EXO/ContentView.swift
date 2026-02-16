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
    @EnvironmentObject private var thunderboltBridgeService: ThunderboltBridgeService
    @EnvironmentObject private var settingsWindowController: SettingsWindowController
    @State private var focusedNode: NodeViewModel?
    @State private var deletingInstanceIDs: Set<String> = []
    @State private var showAllNodes = false
    @State private var showAllInstances = false
    @State private var baseURLCopied = false

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
        // Show warning if local network is not working and EXO is running.
        // The checker uses a longer timeout on first launch to allow time for
        // the permission prompt, so this correctly handles both:
        // 1. User denied permission on first launch
        // 2. Permission broke after restart (macOS TCC bug)
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
                baseURLRow
                Divider()
                    .padding(.vertical, 8)
            } else {
                Divider()
                    .padding(.vertical, 4)
            }
            HoverButton(
                title: "Settings",
                tint: .primary,
                trailingSystemImage: "gear"
            ) {
                settingsWindowController.open(
                    controller: controller,
                    updater: updater,
                    networkStatusService: networkStatusService,
                    thunderboltBridgeService: thunderboltBridgeService,
                    stateService: stateService
                )
            }
            HoverButton(
                title: "Check for Updates",
                tint: .primary,
                trailingSystemImage: "arrow.triangle.2.circlepath"
            ) {
                updater.checkForUpdates()
            }
            .padding(.bottom, 8)
            HoverButton(title: "Quit", tint: .secondary) {
                controller.stop()
                NSApplication.shared.terminate(nil)
            }
        }
    }

    private var dashboardButton: some View {
        HoverButton(
            title: "Web Dashboard",
            tint: .primary,
            trailingSystemImage: "arrow.up.right"
        ) {
            guard let url = URL(string: "http://localhost:52415/") else { return }
            NSWorkspace.shared.open(url)
        }
    }

    private var baseURLRow: some View {
        HStack(spacing: 6) {
            Image(systemName: "link")
                .imageScale(.small)
                .foregroundColor(.secondary)
            Text("localhost:52415/v1")
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.primary)
            Spacer()
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString("http://localhost:52415/v1", forType: .string)
                baseURLCopied = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    baseURLCopied = false
                }
            } label: {
                Image(systemName: baseURLCopied ? "checkmark" : "doc.on.doc")
                    .imageScale(.small)
                    .foregroundColor(baseURLCopied ? .green : .secondary)
                    .contentTransition(.symbolEffect(.replace))
            }
            .buttonStyle(.plain)
            .help("Copy API base URL")
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
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
    let small: Bool
    let action: () -> Void

    init(
        title: String, tint: Color = .primary, trailingSystemImage: String? = nil,
        small: Bool = false, action: @escaping () -> Void
    ) {
        self.title = title
        self.tint = tint
        self.trailingSystemImage = trailingSystemImage
        self.small = small
        self.action = action
    }

    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack {
                Text(title)
                    .font(small ? .caption : nil)
                Spacer()
                if let systemName = trailingSystemImage {
                    Image(systemName: systemName)
                        .imageScale(.small)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, small ? 4 : 6)
            .padding(.horizontal, small ? 6 : 8)
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
