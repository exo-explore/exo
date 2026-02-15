import SwiftUI

/// Native macOS Settings window following Apple HIG.
/// Organized into General, Model, and Advanced sections.
struct SettingsView: View {
    @EnvironmentObject private var controller: ExoProcessController
    @EnvironmentObject private var updater: SparkleUpdater

    @State private var pendingNamespace: String = ""
    @State private var pendingHFToken: String = ""
    @State private var pendingEnableImageModels = false
    @State private var needsRestart = false

    var body: some View {
        TabView {
            generalTab
                .tabItem {
                    Label("General", systemImage: "gear")
                }
            modelTab
                .tabItem {
                    Label("Model", systemImage: "cube")
                }
            aboutTab
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
        }
        .frame(width: 450, height: 320)
        .onAppear {
            pendingNamespace = controller.customNamespace
            pendingHFToken = controller.hfToken
            pendingEnableImageModels = controller.enableImageModels
            needsRestart = false
        }
    }

    // MARK: - General Tab

    private var generalTab: some View {
        Form {
            Section {
                LabeledContent("Cluster Namespace") {
                    TextField("default", text: $pendingNamespace)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 200)
                }
                Text("Nodes with the same namespace form a cluster. Leave empty for default.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section {
                LabeledContent("HuggingFace Token") {
                    SecureField("optional", text: $pendingHFToken)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 200)
                }
                Text("Required for gated models. Get yours at huggingface.co/settings/tokens")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section {
                HStack {
                    Spacer()
                    Button("Save & Restart") {
                        applyGeneralSettings()
                    }
                    .disabled(!hasGeneralChanges)
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - Model Tab

    private var modelTab: some View {
        Form {
            Section {
                Toggle("Enable Image Models (experimental)", isOn: $pendingEnableImageModels)
                Text("Allow text-to-image and image-to-image models in the model picker.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section {
                HStack {
                    Spacer()
                    Button("Save & Restart") {
                        applyModelSettings()
                    }
                    .disabled(!hasModelChanges)
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - About Tab

    private var aboutTab: some View {
        Form {
            Section {
                LabeledContent("Version") {
                    Text(buildTag)
                        .textSelection(.enabled)
                }
                LabeledContent("Commit") {
                    Text(buildCommit)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                }
            }

            Section {
                Button("Check for Updates") {
                    updater.checkForUpdates()
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - Helpers

    private var hasGeneralChanges: Bool {
        pendingNamespace != controller.customNamespace || pendingHFToken != controller.hfToken
    }

    private var hasModelChanges: Bool {
        pendingEnableImageModels != controller.enableImageModels
    }

    private func applyGeneralSettings() {
        controller.customNamespace = pendingNamespace
        controller.hfToken = pendingHFToken
        restartIfRunning()
    }

    private func applyModelSettings() {
        controller.enableImageModels = pendingEnableImageModels
        restartIfRunning()
    }

    private func restartIfRunning() {
        if controller.status == .running || controller.status == .starting {
            controller.restart()
        }
    }

    private var buildTag: String {
        Bundle.main.infoDictionary?["EXOBuildTag"] as? String ?? "unknown"
    }

    private var buildCommit: String {
        Bundle.main.infoDictionary?["EXOBuildCommit"] as? String ?? "unknown"
    }
}
