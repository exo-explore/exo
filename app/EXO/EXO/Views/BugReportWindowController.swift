import AppKit
import SwiftUI

/// Manages a standalone window for the bug-report flow.
/// Ensures only one instance exists and brings it to front on repeated opens.
@MainActor
final class BugReportWindowController: ObservableObject {
    private var window: NSWindow?

    func open() {
        if let existing = window, existing.isVisible {
            existing.makeKeyAndOrderFront(nil)
            NSApp.activate()
            return
        }

        let view = BugReportView(onDismiss: { [weak self] in
            self?.window?.close()
        })

        let hostingController = NSHostingController(rootView: view)
        hostingController.sizingOptions = [.preferredContentSize, .minSize]

        let newWindow = NSWindow(contentViewController: hostingController)
        newWindow.styleMask = [.titled, .closable, .resizable]
        newWindow.title = "Send a Bug Report"
        newWindow.center()
        newWindow.setFrameAutosaveName("ExoBugReportWindow")
        newWindow.isReleasedWhenClosed = false
        newWindow.makeKeyAndOrderFront(nil)
        NSApp.activate()

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
        .frame(minWidth: 380)
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

            Text("Diagnostic logs will be uploaded with your report.")
                .font(.caption)
                .foregroundColor(.secondary)

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
        phase = .sending("Collecting logs and uploading…")
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
