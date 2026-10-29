import AppKit
import SwiftUI

/// Manages a standalone window for the bug-report flow.
/// Ensures only one instance exists and brings it to front on repeated opens.
@MainActor
final class BugReportWindowController: ObservableObject {
    private var window: NSWindow?

    func open() {
        if let existing = window, existing.isVisible {
            WindowActivation.bringToFront(existing)
            return
        }

        let view = BugReportView(onDismiss: { [weak self] in
            self?.window?.close()
        })

        let hostingController = NSHostingController(rootView: view)
        hostingController.sizingOptions = [.preferredContentSize, .minSize]

        let newWindow = NSWindow(contentViewController: hostingController)
        newWindow.styleMask = [.titled, .closable, .resizable]
        newWindow.title = "Report a Bug"
        newWindow.center()
        newWindow.setFrameAutosaveName("ExoBugReportWindow")
        newWindow.isReleasedWhenClosed = false
        WindowActivation.bringToFront(newWindow)

        window = newWindow
    }
}

private struct BugReportView: View {
    fileprivate enum Phase: Equatable {
        case prompting
        case sending(String)
        case success(String)
        case failure(String)
    }

    let onDismiss: () -> Void

    @State private var phase: Phase = .prompting
    @State private var userDescription: String = ""
    @State private var showsDetails = false
    @State private var preparingFiles = false
    @State private var filesError: String?
    @FocusState private var descriptionFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            switch phase {
            case .prompting:
                promptingView
            case .sending(let message):
                sendingView(message: message)
            case .success(let message):
                successView(message: message)
            case .failure(let message):
                failureView(message: message)
            }
        }
        .padding(16)
        .frame(minWidth: 380, idealWidth: 460, maxWidth: .infinity)
        .animation(.easeInOut(duration: 0.2), value: phase)
        .onAppear { descriptionFocused = true }
    }

    private var promptingView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Description (optional)")
                .font(.subheadline)
                .foregroundColor(.secondary)
            ZStack(alignment: .topLeading) {
                if userDescription.isEmpty {
                    Text("What were you doing when it broke?")
                        .font(.body)
                        .foregroundColor(Color(nsColor: .placeholderTextColor))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $userDescription)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(4)
                    .frame(height: 72)
                    .focused($descriptionFocused)
            }
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color(nsColor: .textBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(Color(nsColor: .separatorColor), lineWidth: 1)
            )

            whatIsSent

            HStack {
                Spacer()
                Button("Cancel") { onDismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Send") {
                    Task { await send() }
                }
                .keyboardShortcut(.defaultAction)
            }
            .padding(.top, 4)
        }
    }

    /// What the report includes, with the details under a disclosure and a
    /// way to see the files themselves.
    private var whatIsSent: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(
                "Your report includes EXO's logs, this Mac's model and macOS version, its network setup, and your cluster's status."
            )
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            DisclosureGroup(isExpanded: $showsDetails) {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Self.details, id: \.self) { line in
                        Label {
                            Text(line).fixedSize(horizontal: false, vertical: true)
                        } icon: {
                            Image(systemName: "circle.fill").font(.system(size: 4))
                        }
                    }
                    HStack {
                        Button("Show Report Files…", action: showReportFiles)
                            .disabled(preparingFiles)
                        if preparingFiles {
                            ProgressView().controlSize(.small)
                        }
                    }
                    .padding(.top, 4)
                    if let filesError {
                        Text(filesError)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.callout)
                .padding(.top, 4)
            } label: {
                // The whole label toggles, not only the triangle.
                Text("Details")
                    .contentShape(Rectangle())
                    .onTapGesture {
                        withAnimation { showsDetails.toggle() }
                    }
            }
            .font(.callout)
        }
    }

    static let details = [
        "EXO's logs, with your account name, this Mac's name and your prompts taken out.",
        "This Mac's model, macOS version and time since it started.",
        "Its network interfaces and their addresses, Thunderbolt Bridge and RDMA status. "
            + "Not hardware addresses or your Wi-Fi network.",
        "Your cluster's status and recent activity: its devices, models and errors. "
            + "Not your prompts, the models' responses, or device names.",
        "What you write above.",
    ]

    /// Writes the report to a new temporary folder and shows it in Finder.
    private func showReportFiles() {
        preparingFiles = true
        filesError = nil
        let description = userDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        Task {
            do {
                let files = try await BugReportService().collect(
                    isManual: true, userDescription: description.isEmpty ? nil : description)
                let folder = FileManager.default.temporaryDirectory
                    .appendingPathComponent(
                        "EXO Bug Report \(BugReportService.runTimestampString(Date()))",
                        isDirectory: true)
                try BugReportService.write(files, to: folder)
                NSWorkspace.shared.open(folder)
            } catch {
                filesError = "The report couldn't be prepared: \(error.localizedDescription)"
            }
            preparingFiles = false
        }
    }

    private func sendingView(message: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text(message)
                    .foregroundColor(.secondary)
            }
            HStack {
                Spacer()
                Button("Cancel") { onDismiss() }
                    .keyboardShortcut(.cancelAction)
                    .disabled(true)
                Button("Send") {}
                    .disabled(true)
            }
        }
    }

    private func successView(message: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .font(.title2)
                Text(message)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Button {
                    openGitHubIssue()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.up.right.square")
                        Text("Open GitHub Issue")
                    }
                }
                Spacer()
                Button("Done") { onDismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
    }

    private func failureView(message: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                    .font(.title2)
                Text(message)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("Try Again") {
                    phase = .prompting
                }
                Button("Close") { onDismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
    }

    private func send() async {
        phase = .sending("Sending report…")
        let service = BugReportService()
        let description = userDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            let outcome = try await service.sendReport(
                isManual: true,
                userDescription: description.isEmpty ? nil : description
            )
            if outcome.success {
                phase = .success(outcome.message)
            } else {
                phase = .failure(outcome.message)
            }
        } catch {
            phase = .failure(error.localizedDescription)
        }
    }

    private func openGitHubIssue() {
        let description = userDescription.trimmingCharacters(in: .whitespacesAndNewlines)

        var bodyParts: [String] = []
        bodyParts.append("## Describe the bug")
        bodyParts.append("")
        if !description.isEmpty {
            bodyParts.append(description)
        } else {
            bodyParts.append("A clear and concise description of what the bug is.")
        }
        bodyParts.append("")
        bodyParts.append("## Environment")
        bodyParts.append("")
        bodyParts.append("- macOS Version: \(ProcessInfo.processInfo.operatingSystemVersionString)")
        bodyParts.append("- EXO Version: \(buildTag) (\(buildCommit))")
        bodyParts.append("")
        bodyParts.append("## Additional context")
        bodyParts.append("")
        bodyParts.append("A bug report with diagnostic logs was submitted via the app.")

        let body = bodyParts.joined(separator: "\n")

        var components = URLComponents(string: "https://github.com/exo-explore/exo/issues/new")!
        components.queryItems = [
            URLQueryItem(name: "template", value: "bug_report.md"),
            URLQueryItem(name: "title", value: "[BUG] "),
            URLQueryItem(name: "body", value: body),
            URLQueryItem(name: "labels", value: "bug"),
        ]

        if let url = components.url {
            NSWorkspace.shared.open(url)
        }
    }

    private var buildTag: String {
        Bundle.main.infoDictionary?["EXOBuildTag"] as? String ?? "unknown"
    }

    private var buildCommit: String {
        Bundle.main.infoDictionary?["EXOBuildCommit"] as? String ?? "unknown"
    }
}
