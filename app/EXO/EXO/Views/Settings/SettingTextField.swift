import SwiftUI

/// A text field for a setting exo reads when it starts. What is typed is
/// saved when the user presses Return or leaves the field, not on every
/// keystroke, so exo isn't restarted with a half-typed value.
struct SettingTextField: View {
    let title: String
    @Binding var value: String
    var prompt: Text? = nil
    var isSecure = false
    /// Applied to what was typed before it is saved, for example trimming.
    var normalize: (String) -> String = { $0 }

    @EnvironmentObject private var controller: ExoProcessController
    @State private var draft = ""
    @State private var fieldID = UUID()
    @FocusState private var isFocused: Bool

    var body: some View {
        field
            .focused($isFocused)
            .onSubmit(commit)
            .onAppear { draft = value }
            .onChange(of: value) { _, newValue in
                if !isFocused { draft = newValue }
            }
            .onChange(of: draft) { _, _ in reportEditing() }
            .onChange(of: isFocused) { _, focused in
                if !focused { commit() }
                reportEditing()
            }
            .onDisappear {
                commit()
                controller.settingsField(fieldID, isEditing: false)
            }
    }

    @ViewBuilder private var field: some View {
        if isSecure {
            SecureField(title, text: $draft, prompt: prompt)
        } else {
            TextField(title, text: $draft, prompt: prompt)
        }
    }

    private func commit() {
        let normalized = normalize(draft)
        if normalized != value {
            value = normalized
        }
        draft = normalized
        reportEditing()
    }

    /// Changes elsewhere in Settings wait while this field holds unsaved
    /// typing.
    private func reportEditing() {
        controller.settingsField(fieldID, isEditing: isFocused && normalize(draft) != value)
    }
}
