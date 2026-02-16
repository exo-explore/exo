import AppKit
import SwiftUI

/// A small floating callout that drops down from the menu bar area on first launch,
/// pointing the user to the web dashboard. Clean, minimal, speech-bubble style.
@MainActor
final class FirstLaunchPopout {
    private var panel: NSPanel?
    private var countdownTask: Task<Void, Never>?
    private static let dashboardURL = "http://localhost:52415/"

    func show() {
        guard panel == nil else { return }

        let hostingView = NSHostingView(
            rootView: WelcomeCalloutView(
                onDismiss: { [weak self] in
                    self?.dismiss()
                },
                onOpen: { [weak self] in
                    self?.openDashboard()
                    self?.dismiss()
                }))
        hostingView.frame = NSRect(x: 0, y: 0, width: 280, height: 100)

        let window = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 280, height: 100),
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
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
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true

        // Position near top-right, just below the menu bar
        if let screen = NSScreen.main {
            let screenFrame = screen.visibleFrame
            let x = screenFrame.maxX - window.frame.width - 16
            let y = screenFrame.maxY - 8
            window.setFrameOrigin(NSPoint(x: x, y: y))
        }

        window.alphaValue = 0
        window.orderFrontRegardless()
        panel = window

        // Fade in
        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.3
            context.timingFunction = CAMediaTimingFunction(name: .easeOut)
            window.animator().alphaValue = 1
        }

        // Auto-open dashboard after 5s then dismiss
        countdownTask = Task {
            try? await Task.sleep(nanoseconds: 5_000_000_000)
            if !Task.isCancelled {
                openDashboard()
                try? await Task.sleep(nanoseconds: 800_000_000)
                if !Task.isCancelled {
                    dismiss()
                }
            }
        }
    }

    func dismiss() {
        countdownTask?.cancel()
        countdownTask = nil
        guard let window = panel else { return }
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.2
            context.timingFunction = CAMediaTimingFunction(name: .easeIn)
            window.animator().alphaValue = 0
        }, completionHandler: {
            Task { @MainActor in
                window.close()
            }
        })
        panel = nil
    }

    private func openDashboard() {
        guard let url = URL(string: Self.dashboardURL) else { return }
        NSWorkspace.shared.open(url)
    }
}

/// Minimal welcome callout — friendly pointer, not a wall of text.
private struct WelcomeCalloutView: View {
    let onDismiss: () -> Void
    let onOpen: () -> Void
    @State private var countdown = 5
    @State private var timerTask: Task<Void, Never>?
    @State private var appeared = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                Text("Welcome to EXO!")
                    .font(.system(.headline, design: .rounded))
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                Spacer()
                Button {
                    onDismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }

            Text("Run your first model here:")
                .font(.system(.subheadline, design: .default))
                .foregroundColor(.secondary)

            HStack {
                Button {
                    onOpen()
                } label: {
                    Label("Open Dashboard", systemImage: "arrow.up.right.square")
                        .font(.system(.caption, design: .default))
                        .fontWeight(.medium)
                }
                .buttonStyle(.borderedProminent)
                .tint(.accentColor)
                .controlSize(.small)

                Spacer()

                if countdown > 0 {
                    Text("Opening in \(countdown)s")
                        .font(.system(.caption2, design: .default))
                        .foregroundColor(.secondary.opacity(0.6))
                        .monospacedDigit()
                }
            }
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(.ultraThinMaterial)
                .shadow(color: .black.opacity(0.15), radius: 12, y: 4)
        }
        .padding(4)
        .opacity(appeared ? 1 : 0)
        .offset(y: appeared ? 0 : -8)
        .onAppear {
            withAnimation(.easeOut(duration: 0.35).delay(0.05)) {
                appeared = true
            }
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
