import AppKit
import SwiftUI

/// The welcome window's panes. Standard controls only; the brand shows
/// through the app icon and the menu bar glyph.
struct WelcomeView: View {
    @ObservedObject var model: WelcomeModel

    var body: some View {
        VStack(spacing: 0) {
            switch model.flow.pane {
            case .welcome:
                welcomePane
            case .findMacs:
                findMacsPane
            case .ready:
                readyPane
            }
        }
        .frame(width: 480)
        .animation(.default, value: model.flow)
    }

    // MARK: - Welcome

    private var welcomePane: some View {
        pane(
            header: {
                Image(nsImage: NSApp.applicationIconImage)
                    .resizable()
                    .frame(width: 96, height: 96)
                    .accessibilityHidden(true)
            },
            title: "Welcome to EXO",
            message: "Run AI models on this Mac, and add your other Macs to run bigger ones.",
            content: {
                Toggle("Open EXO at login", isOn: $model.openAtLogin)
                    .toggleStyle(.checkbox)
                    .help(
                        "EXO starts quietly in the menu bar when you log in, so this Mac stays in your cluster."
                    )
            },
            buttons: {
                Button("Continue") { model.send(.continueFromWelcome) }
                    .keyboardShortcut(.defaultAction)
            })
    }

    // MARK: - Find your other Macs

    private var findMacsPane: some View {
        pane(
            header: {
                Image(systemName: "network")
                    .font(.system(size: 52, weight: .light))
                    .foregroundStyle(.secondary)
                    .frame(height: 96)
                    .accessibilityHidden(true)
            },
            title: "Find your other Macs",
            message:
                "EXO looks for other Macs running EXO on your network so they can share the work. macOS will ask you to allow this.",
            content: { findMacsDetail },
            buttons: { findMacsButtons })
    }

    @ViewBuilder
    private var findMacsDetail: some View {
        switch model.flow.localNetwork {
        case .notAsked, .allowed:
            Text(
                "Only using this Mac? Allow it anyway: nothing is shared until another of your Macs joins."
            )
            .font(.callout)
            .foregroundStyle(.secondary)
        case .asking:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Waiting for macOS…")
                    .foregroundStyle(.secondary)
            }
        case .denied:
            Label {
                Text(
                    "Local Network access is off, so EXO works on this Mac only. To add Macs later, turn on EXO in Privacy & Security › Local Network."
                )
            } icon: {
                Image(systemName: "exclamationmark.triangle")
            }
            .font(.callout)
        }
    }

    @ViewBuilder
    private var findMacsButtons: some View {
        switch model.flow.localNetwork {
        case .notAsked, .allowed:
            Button("Continue") { model.send(.askForLocalNetwork) }
                .keyboardShortcut(.defaultAction)
        case .asking:
            Button("Continue") {}
                .disabled(true)
        case .denied:
            Button("Open Privacy & Security…") { model.openPrivacySettings() }
            Button("Continue") { model.send(.continueWithoutLocalNetwork) }
                .keyboardShortcut(.defaultAction)
        }
    }

    // MARK: - Ready

    private var readyPane: some View {
        pane(
            header: { readyHeader },
            title: readyTitle,
            message: readyMessage,
            content: { menuBarHint },
            buttons: { readyButtons })
    }

    @ViewBuilder
    private var readyHeader: some View {
        Group {
            switch model.exoStatus {
            case .running:
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 52))
                    .foregroundStyle(.green)
            case .failed:
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 52))
                    .foregroundStyle(.orange)
            case .starting, .stopped:
                ProgressView()
                    .controlSize(.large)
            }
        }
        .frame(height: 96)
        .accessibilityHidden(true)
    }

    private var readyTitle: String {
        switch model.exoStatus {
        case .running: return "EXO is running"
        case .failed: return "EXO couldn't start"
        case .starting, .stopped: return "Starting EXO…"
        }
    }

    private var readyMessage: String {
        switch model.exoStatus {
        case .running:
            return "Choose a model and start chatting in the dashboard, in your browser."
        case .failed(let reason):
            return reason
        case .starting, .stopped:
            return "This takes a few seconds."
        }
    }

    private var menuBarHint: some View {
        HStack(spacing: 6) {
            Text("EXO lives in the menu bar")
            Image(nsImage: MenuBarGlyph.image(style: Self.glyphStyle, alpha: 1))
                .renderingMode(.template)
                .foregroundStyle(.primary)
                .accessibilityLabel("EXO menu bar icon")
            Text("— click it any time.")
        }
        .font(.callout)
        .foregroundStyle(.secondary)
    }

    private static let glyphStyle = MenuBarIconStyle.showing(
        .running, apiState: .responding, problemInMenu: false, updateAvailable: false)

    @ViewBuilder
    private var readyButtons: some View {
        if case .failed = model.exoStatus {
            Button("Try Again") { model.restartExo() }
        }
        Button("Open Dashboard") { model.openDashboard() }
            .keyboardShortcut(.defaultAction)
            .disabled(model.exoStatus != .running)
    }

    // MARK: - Layout

    private func pane<Header: View, Content: View, Buttons: View>(
        @ViewBuilder header: () -> Header, title: String, message: String,
        @ViewBuilder content: () -> Content, @ViewBuilder buttons: () -> Buttons
    ) -> some View {
        VStack(spacing: 0) {
            VStack(spacing: 12) {
                header()
                Text(title)
                    .font(.title2.weight(.semibold))
                Text(message)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                content()
                    .padding(.top, 4)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 40)
            .padding(.top, 28)
            .padding(.bottom, 20)
            HStack {
                Spacer()
                buttons()
            }
            .padding(20)
        }
    }
}
