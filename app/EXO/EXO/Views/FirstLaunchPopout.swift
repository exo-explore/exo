import AppKit
import SwiftUI

/// A small floating panel that appears near the menu bar on first launch,
/// showing a countdown before auto-opening the dashboard.
/// Inspired by LlamaBarn's menu bar popout pattern.
@MainActor
final class FirstLaunchPopout {
    private var panel: NSPanel?
    private var countdownTask: Task<Void, Never>?
    private static let dashboardURL = "http://localhost:52415/"

    func show() {
        guard panel == nil else { return }

        let hostingView = NSHostingView(
            rootView: PopoutContentView(
                onDismiss: { [weak self] in
                    self?.dismiss()
                },
                onOpen: { [weak self] in
                    self?.openDashboard()
                }))
        hostingView.frame = NSRect(x: 0, y: 0, width: 300, height: 120)

        let window = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 300, height: 120),
            styleMask: [.nonactivatingPanel, .hudWindow, .utilityWindow],
            backing: .buffered,
            defer: false
        )
        window.contentView = hostingView
        window.isFloatingPanel = true
        window.level = .floating
        window.hasShadow = true
        window.isOpaque = false
        window.backgroundColor = .clear
        window.isMovableByWindowBackground = false
        window.hidesOnDeactivate = false
        window.collectionBehavior = [.canJoinAllSpaces, .stationary]

        // Position near top-right of screen (near menu bar area)
        if let screen = NSScreen.main {
            let screenFrame = screen.visibleFrame
            let x = screenFrame.maxX - window.frame.width - 16
            let y = screenFrame.maxY - 8
            window.setFrameOrigin(NSPoint(x: x, y: y))
        }

        window.orderFrontRegardless()
        panel = window

        // Start countdown: auto-open dashboard after 5 seconds, then dismiss
        countdownTask = Task {
            try? await Task.sleep(nanoseconds: 5_000_000_000)
            if !Task.isCancelled {
                openDashboard()
                // Give the browser a moment, then dismiss
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if !Task.isCancelled {
                    dismiss()
                }
            }
        }
    }

    func dismiss() {
        countdownTask?.cancel()
        countdownTask = nil
        panel?.close()
        panel = nil
    }

    private func openDashboard() {
        guard let url = URL(string: Self.dashboardURL) else { return }
        NSWorkspace.shared.open(url)
    }
}

/// SwiftUI content for the first-launch popout
private struct PopoutContentView: View {
    let onDismiss: () -> Void
    let onOpen: () -> Void
    @State private var countdown = 5
    @State private var timerTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .imageScale(.large)
                Text("EXO is ready!")
                    .font(.system(.headline, design: .default))
                    .fontWeight(.semibold)
                Spacer()
                Button {
                    onDismiss()
                } label: {
                    Image(systemName: "xmark")
                        .imageScale(.small)
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }

            Text("http://localhost:52415")
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.secondary)

            HStack {
                Button {
                    onOpen()
                    onDismiss()
                } label: {
                    Text("Open Dashboard")
                        .font(.caption)
                        .fontWeight(.medium)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)

                Spacer()

                Text("Opening in \(countdown)s")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(12)
        .onAppear {
            startCountdown()
        }
        .onDisappear {
            timerTask?.cancel()
            timerTask = nil
        }
    }

    private func startCountdown() {
        timerTask = Task {
            while countdown > 0 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if !Task.isCancelled {
                    countdown -= 1
                }
            }
        }
    }
}
