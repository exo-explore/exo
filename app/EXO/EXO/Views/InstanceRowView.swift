import SwiftUI

struct InstanceRowView: View {
    let instance: InstanceViewModel
    @State private var animatedTaskIDs: Set<String> = []
    @State private var infoTask: InstanceTaskViewModel?
    @State private var showChatTasks = true

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(instance.modelName)
                        .font(.subheadline)
                    Text(instance.nodeSummary)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                if let progress = instance.downloadProgress {
                    downloadStatusView(progress: progress)
                } else {
                    statusChip(label: instance.state.label.uppercased(), color: statusColor)
                }
            }
            if let progress = instance.downloadProgress {
                GeometryReader { geometry in
                    HStack {
                        Spacer()
                        downloadProgressBar(progress: progress)
                            .frame(width: geometry.size.width * 0.5)
                    }
                }
                .frame(height: 4)
                .padding(.top, -8)
                .padding(.bottom, 2)
                HStack(spacing: 8) {
                    Text(instance.sharding ?? "")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Spacer()
                    downloadSpeedView(progress: progress)
                }
            } else {
                Text(instance.sharding ?? "")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            if !instance.chatTasks.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Chat Tasks")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("(\(instance.chatTasks.count))")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer()
                        collapseButton(isExpanded: $showChatTasks)
                    }
                    .animation(nil, value: showChatTasks)
                    if showChatTasks {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(instance.chatTasks) { task in
                                taskRow(for: task, parentModelName: instance.modelName)
                            }
                        }
                        .transition(.opacity)
                    }
                }
                .padding(.top, 4)
                .animation(.easeInOut(duration: 0.25), value: showChatTasks)
            }
        }
        .padding(.vertical, 6)
    }

    private var statusColor: Color {
        switch instance.state {
        case .downloading: return .blue
        case .warmingUp: return .orange
        case .running: return .green
        case .ready: return .teal
        case .waiting, .idle: return .gray
        case .failed: return .red
        case .unknown: return .secondary
        }
    }

    @ViewBuilder
    private func taskRow(for task: InstanceTaskViewModel, parentModelName: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .top, spacing: 8) {
                taskStatusIcon(for: task)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Chat")
                        .font(.caption)
                        .fontWeight(.semibold)
                    if let subtitle = task.subtitle,
                        subtitle.caseInsensitiveCompare(parentModelName) != .orderedSame
                    {
                        Text(subtitle)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    if let prompt = task.promptPreview, !prompt.isEmpty {
                        Text("⊙ \(prompt)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .lineLimit(2)
                    }
                    if task.status == .failed, let error = task.errorMessage, !error.isEmpty {
                        Text(error)
                            .font(.caption2)
                            .foregroundColor(.red)
                            .lineLimit(3)
                    }
                }
                Spacer(minLength: 6)
                Button {
                    infoTask = task
                } label: {
                    Image(systemName: "info.circle")
                        .imageScale(.small)
                }
                .buttonStyle(.plain)
                .popover(
                    item: Binding<InstanceTaskViewModel?>(
                        get: { infoTask?.id == task.id ? infoTask : nil },
                        set: { newValue in
                            if newValue == nil {
                                infoTask = nil
                            } else {
                                infoTask = newValue
                            }
                        }
                    ),
                    attachmentAnchor: .rect(.bounds),
                    arrowEdge: .top
                ) { _ in
                    TaskDetailView(task: task)
                        .padding()
                        .frame(width: 240)
                }
            }
        }
    }

    private func taskStatusIcon(for task: InstanceTaskViewModel) -> some View {
        let icon: String
        let color: Color
        let animation: Animation?

        switch task.status {
        case .running:
            icon = "arrow.triangle.2.circlepath"
            color = .blue
            animation = Animation.linear(duration: 1).repeatForever(autoreverses: false)
        case .pending:
            icon = "circle.dashed"
            color = .secondary
            animation = nil
        case .failed:
            icon = "exclamationmark.triangle.fill"
            color = .red
            animation = nil
        case .complete:
            icon = "checkmark.circle.fill"
            color = .green
            animation = nil
        case .unknown:
            icon = "questionmark.circle"
            color = .secondary
            animation = nil
        }

        let image = Image(systemName: icon)
            .imageScale(.small)
            .foregroundColor(color)

        if let animation {
            return AnyView(
                image
                    .rotationEffect(.degrees(animatedTaskIDs.contains(task.id) ? 360 : 0))
                    .onAppear {
                        if !animatedTaskIDs.contains(task.id) {
                            animatedTaskIDs.insert(task.id)
                        }
                    }
                    .animation(animation, value: animatedTaskIDs)
            )
        }

        return AnyView(image)
    }

    private func statusChip(label: String, color: Color) -> some View {
        Text(label)
            .font(.caption2)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color.opacity(0.15))
            .foregroundColor(color)
            .clipShape(Capsule())
    }

    private func downloadStatusView(progress: DownloadProgressViewModel) -> some View {
        VStack(alignment: .trailing, spacing: 4) {
            statusChip(label: "DOWNLOADING", color: .blue)
            Text(progress.formattedProgress)
                .foregroundColor(.primary)
        }
        .font(.caption2)
    }

    private func downloadSpeedView(progress: DownloadProgressViewModel) -> some View {
        HStack(spacing: 4) {
            Text(progress.formattedSpeed)
            if let eta = progress.formattedETA {
                Text("·")
                Text(eta)
            }
        }
        .font(.caption2)
        .foregroundColor(.secondary)
    }

    private func downloadProgressBar(progress: DownloadProgressViewModel) -> some View {
        ProgressView(value: progress.fractionCompleted)
            .progressViewStyle(.linear)
            .tint(.blue)
    }

    private func collapseButton(isExpanded: Binding<Bool>) -> some View {
        Button {
            isExpanded.wrappedValue.toggle()
        } label: {
            Label(
                isExpanded.wrappedValue ? "Hide" : "Show",
                systemImage: isExpanded.wrappedValue ? "chevron.up" : "chevron.down"
            )
            .labelStyle(.titleAndIcon)
            .contentTransition(.symbolEffect(.replace))
        }
        .buttonStyle(.plain)
        .font(.caption2)
    }

    private struct TaskDetailView: View, Identifiable {
        let task: InstanceTaskViewModel
        var id: String { task.id }

        var body: some View {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    parameterSection
                    messageSection
                    if let error = task.errorMessage, !error.isEmpty {
                        detailRow(
                            icon: "exclamationmark.triangle.fill",
                            title: "Error",
                            value: error,
                            tint: .red
                        )
                    }
                }
            }
        }

        @ViewBuilder
        private var parameterSection: some View {
            if let params = task.parameters {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Parameters")
                        .font(.subheadline)
                    if let temperature = params.temperature {
                        detailRow(title: "Temperature", value: String(format: "%.1f", temperature))
                    }
                    if let maxTokens = params.maxTokens {
                        detailRow(title: "Max Tokens", value: "\(maxTokens)")
                    }
                    if let stream = params.stream {
                        detailRow(title: "Stream", value: stream ? "On" : "Off")
                    }
                    if let topP = params.topP {
                        detailRow(title: "Top P", value: String(format: "%.2f", topP))
                    }
                }
            }
        }

        @ViewBuilder
        private var messageSection: some View {
            if let messages = task.parameters?.messages, !messages.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Messages")
                        .font(.subheadline)
                    ForEach(Array(messages.enumerated()), id: \.offset) { _, message in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(message.role?.capitalized ?? "Message")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            if let content = message.content, !content.isEmpty {
                                Text(content)
                                    .font(.caption2)
                                    .foregroundColor(.primary)
                            }
                        }
                        .padding(8)
                        .background(Color.secondary.opacity(0.08))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                }
            }
        }

        @ViewBuilder
        private func detailRow(
            icon: String? = nil, title: String, value: String, tint: Color = .secondary
        ) -> some View {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                if let icon {
                    Image(systemName: icon)
                        .imageScale(.small)
                        .foregroundColor(tint)
                }
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text(value)
                    .font(.caption2)
                    .foregroundColor(.primary)
            }
        }
    }
}
