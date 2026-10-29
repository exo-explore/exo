import AppKit
import SwiftUI

/// Settings › Advanced: performance, environment variables and diagnostics.
struct AdvancedSettingsPane: View {
    @EnvironmentObject private var controller: ExoProcessController

    @State private var selectedVariables = Set<CustomEnvironmentVariable.ID>()
    @State private var showsDiagnostics = false

    var body: some View {
        Form {
            Section {
                Picker("Fast GPU synchronization", selection: $controller.fastSynch) {
                    ForEach(FastSynchSetting.allCases) { setting in
                        Text(setting.title).tag(setting)
                    }
                }
                .help("Passed to exo as EXO_FAST_SYNCH. Automatic passes nothing, so exo decides.")
            } header: {
                Text("Performance")
            } footer: {
                SettingFootnote(
                    "Automatic turns it on only for models that run over RDMA, where it "
                        + "lowers latency. On for every model, it can stall models that share "
                        + "this Mac's GPU.")
            }

            Section {
                environmentVariablesTable
            } header: {
                Text("Environment variables")
            } footer: {
                SettingFootnote(
                    "Passed to EXO when it starts. They override the settings in the other panes.")
            }

            Section {
                DisclosureGroup(isExpanded: $showsDiagnostics) {
                    DiagnosticsView()
                } label: {
                    // The whole label toggles, not only the triangle.
                    Text("Diagnostics")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            withAnimation { showsDiagnostics.toggle() }
                        }
                }
            }

            Section {
                LabeledContent {
                    Button("Show Again") {
                        guard let url = URL(string: "http://localhost:52415/?reset-onboarding")
                        else { return }
                        NSWorkspace.shared.open(url)
                    }
                    .disabled(controller.status != .running)
                } label: {
                    Text("Dashboard onboarding")
                    Text("Opens the dashboard and starts its onboarding again.")
                }
            }
        }
        .settingsPane()
    }

    // MARK: Environment variables

    private var invalidNames: [String] {
        controller.customEnvironmentVariables.filter(\.hasInvalidName).map(\.trimmedKey)
    }

    private var environmentVariablesTable: some View {
        VStack(alignment: .leading, spacing: 6) {
            Table(controller.customEnvironmentVariables, selection: $selectedVariables) {
                TableColumn("Name") { variable in
                    HStack(spacing: 4) {
                        SettingTextField(
                            title: "Name",
                            value: $controller.customEnvironmentVariables.field(
                                \.key, of: variable.id),
                            prompt: Text("NAME"),
                            normalize: { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        )
                        .labelsHidden()
                        .font(.system(.body, design: .monospaced))
                        if variable.hasInvalidName {
                            Image(systemName: "exclamationmark.triangle")
                                .help("Not a valid name, so it isn't passed to EXO.")
                                .accessibilityLabel("Invalid name")
                        }
                    }
                }
                TableColumn("Value") { variable in
                    SettingTextField(
                        title: "Value",
                        value: $controller.customEnvironmentVariables.field(
                            \.value, of: variable.id),
                        prompt: Text("value")
                    )
                    .labelsHidden()
                    .font(.system(.body, design: .monospaced))
                }
            }
            .tableStyle(.bordered(alternatesRowBackgrounds: false))
            .frame(height: 110)

            HStack(spacing: 0) {
                Button {
                    controller.customEnvironmentVariables.append(CustomEnvironmentVariable())
                } label: {
                    Image(systemName: "plus")
                        .frame(width: 22, height: 18)
                }
                .help("Add a variable")
                .accessibilityLabel("Add a variable")
                Divider()
                    .frame(height: 14)
                Button {
                    controller.customEnvironmentVariables.removeAll {
                        selectedVariables.contains($0.id)
                    }
                    selectedVariables.removeAll()
                } label: {
                    Image(systemName: "minus")
                        .frame(width: 22, height: 18)
                }
                .disabled(selectedVariables.isEmpty)
                .help("Remove the selected variables")
                .accessibilityLabel("Remove the selected variables")
                Spacer()
            }
            .buttonStyle(.borderless)

            if let hint = Self.invalidNamesHint(invalidNames) {
                Label(hint, systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// Says which names are invalid, and that they are skipped rather than
    /// blocking the other settings.
    static func invalidNamesHint(_ names: [String]) -> String? {
        guard !names.isEmpty else { return nil }
        let list = names.map { "“\($0)”" }.joined(separator: ", ")
        let isOne = names.count == 1
        return
            "\(list) \(isOne ? "isn't a valid name, so it isn't" : "aren't valid names, so they aren't") "
            + "passed to EXO. Names use letters, digits and underscores, and don't start with a digit."
    }
}
