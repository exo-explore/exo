import AppKit
import SwiftUI

/// A popover callout anchored to the menu bar icon on every launch,
/// pointing the user to the web dashboard with an arrow connecting to the icon.
@MainActor
final class FirstLaunchPopout {
    private var popover: NSPopover?
    private var countdownTask: Task<Void, Never>?
    private static let dashboardURL = "http://localhost:52415/"

    /// Called when the user completes onboarding (clicks Open Dashboard or dismisses).
    var onComplete: (() -> Void)?

    func show() {
        guard popover == nil else { return }

        // The status bar button may not exist yet on first launch; retry generously.
        showWithRetry(attemptsRemaining: 15)
    }

    private func showWithRetry(attemptsRemaining: Int) {
        guard attemptsRemaining > 0 else {
            // Exhausted retries — fall back to just opening the dashboard directly.
            openDashboard()
            onComplete?()
            return
        }

        guard let button = Self.findStatusItemButton() else {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                self?.showWithRetry(attemptsRemaining: attemptsRemaining - 1)
            }
            return
        }

        let pop = NSPopover()
        pop.behavior = .applicationDefined
        pop.animates = true
        pop.contentSize = NSSize(width: 280, height: 120)
        pop.contentViewController = NSHostingController(
            rootView: WelcomeCalloutView(
                countdownDuration: 10,
                onDismiss: { [weak self] in
                    self?.onComplete?()
                    self?.dismiss()
                },
                onOpen: { [weak self] in
                    self?.openDashboard()
                    self?.onComplete?()
                    self?.dismiss()
                }
            )
        )

        self.popover = pop
        pop.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)

        // Auto-open dashboard after 10s then dismiss
        countdownTask = Task {
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            if !Task.isCancelled {
                openDashboard()
                onComplete?()
                dismiss()
            }
        }
    }

    func dismiss() {
        countdownTask?.cancel()
        countdownTask = nil
        guard let pop = popover else { return }
        popover = nil
        pop.performClose(nil)
    }

    private func openDashboard() {
        guard let url = URL(string: Self.dashboardURL) else { return }
        NSWorkspace.shared.open(url)
    }

    /// Finds the NSStatusBarButton created by SwiftUI's MenuBarExtra.
    /// Walks the view hierarchy to find the actual button rather than the content view.
    private static func findStatusItemButton() -> NSView? {
        for window in NSApp.windows {
            let className = NSStringFromClass(type(of: window))
            // Match NSStatusBarWindow or any internal SwiftUI status bar window
            guard className.contains("StatusBar") || className.contains("MenuBarExtra") else {
                continue
            }
            if let content = window.contentView {
                if let button = findButton(in: content) {
                    return button
                }
                // Fall back to the content view itself if it has a non-zero frame
                if content.frame.width > 0 {
                    return content
                }
            }
        }
        return nil
    }

    /// Recursively searches the view hierarchy for an NSStatusBarButton.
    private static func findButton(in view: NSView) -> NSView? {
        let className = NSStringFromClass(type(of: view))
        if className.contains("StatusBarButton") || className.contains("StatusItem") {
            return view
        }
        for subview in view.subviews {
            if let found = findButton(in: subview) {
                return found
            }
        }
        return nil
    }
}

/// Minimal welcome callout — friendly pointer, not a wall of text.
/// Rendered inside the NSPopover which provides its own chrome and arrow.
private struct WelcomeCalloutView: View {
    let countdownDuration: Int
    let onDismiss: () -> Void
    let onOpen: () -> Void
    @State private var countdown: Int
    @State private var timerTask: Task<Void, Never>?

    init(countdownDuration: Int, onDismiss: @escaping () -> Void, onOpen: @escaping () -> Void) {
        self.countdownDuration = countdownDuration
        self.onDismiss = onDismiss
        self.onOpen = onOpen
        self._countdown = State(initialValue: countdownDuration)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                Text("EXO is running")
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
                    Text("Launching in \(countdown) secs...")
                        .font(.system(.caption2, design: .default))
                        .foregroundColor(.secondary.opacity(0.6))
                        .monospacedDigit()
                }
            }
        }
        .padding(14)
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
