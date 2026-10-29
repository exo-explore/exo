import SwiftUI

/// The panes of the Settings window, in toolbar order.
enum SettingsPane: String {
    case general
    case cluster
    case models
    case advanced

    /// Where the last pane shown is kept, so Settings reopens on it.
    static let storageKey = "EXOSettingsPane"
}

extension View {
    /// The layout every Settings pane shares: a grouped form as wide as the
    /// window, as tall as its content, with the restart bar below it while
    /// exo needs restarting to use changed settings.
    func settingsPane() -> some View {
        modifier(SettingsPaneLayout())
    }
}

private struct SettingsPaneLayout: ViewModifier {
    func body(content: Content) -> some View {
        content
            .formStyle(.grouped)
            .scrollDisabled(true)
            .frame(width: SettingsLayout.width)
            .fixedSize(horizontal: false, vertical: true)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                SettingsRestartBar()
            }
    }
}

enum SettingsLayout {
    static let width: CGFloat = 540
}

/// "Restart EXO to use the new settings." Shown in every pane while exo runs
/// with older settings and restarting it would stop models on this Mac.
/// (When nothing would stop, changes restart exo by themselves.)
struct SettingsRestartBar: View {
    @EnvironmentObject private var controller: ExoProcessController

    var body: some View {
        if controller.needsRestart {
            VStack(spacing: 0) {
                Divider()
                HStack(alignment: .center, spacing: 10) {
                    Image(systemName: "arrow.clockwise.circle")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                    Text("Restart EXO to use the new settings. Running models will stop.")
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 8)
                    Button("Restart Now") {
                        controller.restartToApplySettings()
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
            }
            .background(.bar)
        }
    }
}

/// An explanation under a setting, in the style of a form's footnotes.
struct SettingFootnote: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(text)
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}
