import AppKit
import SwiftUI

/// Settings › Models: where models are kept and how they are downloaded.
struct ModelsSettingsPane: View {
    @EnvironmentObject private var controller: ExoProcessController

    @State private var selectedFolders = Set<ModelFolder.ID>()

    var body: some View {
        Form {
            Section("Downloads") {
                downloadFolderRow
                Toggle("Use only downloaded models (offline)", isOn: $controller.offlineMode)
                    .help("Passed to exo as EXO_OFFLINE.")
            }

            Section {
                otherFoldersTable
            } header: {
                Text("Other model folders")
            } footer: {
                SettingFootnote(
                    "EXO also uses the models in these folders. It never downloads into, "
                        + "or deletes from, a read-only folder.")
            }

            Section {
                SettingTextField(
                    title: "Hugging Face access token",
                    value: $controller.hfToken,
                    prompt: Text("Optional"),
                    isSecure: true,
                    normalize: { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                )
                .help("Passed to exo as HF_TOKEN.")
                SettingTextField(
                    title: "Download mirror",
                    value: $controller.hfEndpoint,
                    prompt: Text("huggingface.co"),
                    // "hf-mirror.com" is saved as "https://hf-mirror.com", and
                    // the field shows what is saved.
                    normalize: HuggingFaceEndpoint.normalized
                )
                .help("Passed to exo as HF_ENDPOINT.")
            } footer: {
                // One literal, so the link is parsed as Markdown.
                Text(
                    "A token is needed for gated models, and is kept in your keychain. [Get a token…](https://huggingface.co/settings/tokens) Use a mirror, such as https://hf-mirror.com, if huggingface.co is slow or blocked where you are."
                )
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }

            Section("Experimental") {
                Toggle("Show image generation models", isOn: $controller.enableImageModels)
                    .help("Passed to exo as EXO_ENABLE_IMAGE_MODELS.")
            }
        }
        .settingsPane()
    }

    // MARK: Download folder

    /// exo's default download folder, used when none is set.
    static let defaultDownloadFolder = ExoProcessController.exoDirectoryURL
        .appendingPathComponent("models", isDirectory: true)

    private var downloadFolderURL: URL {
        let path = controller.defaultModelsDir.trimmingCharacters(in: .whitespaces)
        guard !path.isEmpty else { return Self.defaultDownloadFolder }
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
    }

    private var downloadFolderRow: some View {
        LabeledContent {
            HStack {
                if !controller.defaultModelsDir.isEmpty {
                    Button("Use Default") { controller.defaultModelsDir = "" }
                        .help(
                            "Download to \(Self.displayPath(Self.defaultDownloadFolder.path)) again."
                        )
                }
                Button("Choose…", action: chooseDownloadFolder)
                Button("Show in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([downloadFolderURL])
                }
                .disabled(!FileManager.default.fileExists(atPath: downloadFolderURL.path))
            }
        } label: {
            Text("Download models to")
            Text(Self.displayPath(downloadFolderURL.path))
                .truncationMode(.middle)
                .help(downloadFolderURL.path)
        }
        .help("Passed to exo as EXO_DEFAULT_MODELS_DIR.")
    }

    private func chooseDownloadFolder() {
        Task {
            let chosen = await FolderPicker.choose(
                message: "Choose the folder EXO downloads models to.",
                startingAt: downloadFolderURL,
                allowsMultipleSelection: false)
            guard let url = chosen.first else { return }
            // The default folder is kept as "not set", so it follows exo's own
            // default.
            controller.defaultModelsDir =
                url.standardizedFileURL == Self.defaultDownloadFolder.standardizedFileURL
                ? "" : url.path
        }
    }

    // MARK: Other folders

    private var otherFoldersTable: some View {
        VStack(alignment: .leading, spacing: 6) {
            Table(controller.otherModelFolders, selection: $selectedFolders) {
                TableColumn("Folder") { folder in
                    HStack(spacing: 4) {
                        Text(Self.displayPath(folder.path))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        if !folder.isUsable {
                            Image(systemName: "exclamationmark.triangle")
                                .accessibilityLabel("Can't be used")
                        }
                    }
                    .help(
                        folder.isUsable
                            ? folder.path
                            : "EXO can't use a folder whose path contains a colon (:).")
                }
                TableColumn("Read-only") { folder in
                    Toggle("Read-only", isOn: readOnlyBinding(for: folder.id))
                        .toggleStyle(.checkbox)
                        .labelsHidden()
                }
                .width(70)
            }
            .tableStyle(.bordered(alternatesRowBackgrounds: false))
            .frame(height: 110)
            .contextMenu(forSelectionType: ModelFolder.ID.self) { ids in
                Button("Show in Finder") { reveal(ids) }
                Button("Remove") { remove(ids) }
            }

            HStack(spacing: 0) {
                Button {
                    addFolders()
                } label: {
                    Image(systemName: "plus")
                        .frame(width: 22, height: 18)
                }
                .help("Add folders")
                .accessibilityLabel("Add folders")
                Divider()
                    .frame(height: 14)
                Button {
                    remove(selectedFolders)
                } label: {
                    Image(systemName: "minus")
                        .frame(width: 22, height: 18)
                }
                .disabled(selectedFolders.isEmpty)
                .help("Remove the selected folders")
                .accessibilityLabel("Remove the selected folders")
                Spacer()
            }
            .buttonStyle(.borderless)
        }
    }

    private func readOnlyBinding(for id: ModelFolder.ID) -> Binding<Bool> {
        Binding(
            get: { controller.otherModelFolders.first { $0.id == id }?.readOnly ?? false },
            set: { readOnly in
                guard let index = controller.otherModelFolders.firstIndex(where: { $0.id == id })
                else { return }
                controller.otherModelFolders[index].readOnly = readOnly
            }
        )
    }

    private func addFolders() {
        Task {
            let chosen = await FolderPicker.choose(
                message: "Choose folders where EXO also looks for models.",
                startingAt: nil,
                allowsMultipleSelection: true)
            let existing = Set(controller.otherModelFolders.map(\.trimmedPath))
            let added = chosen.map(\.path).filter { !existing.contains($0) }
            controller.otherModelFolders += added.map { ModelFolder(path: $0) }
        }
    }

    private func remove(_ ids: Set<ModelFolder.ID>) {
        controller.otherModelFolders.removeAll { ids.contains($0.id) }
        selectedFolders.subtract(ids)
    }

    private func reveal(_ ids: Set<ModelFolder.ID>) {
        let urls = controller.otherModelFolders
            .filter { ids.contains($0.id) }
            .map { URL(fileURLWithPath: ($0.trimmedPath as NSString).expandingTildeInPath) }
        NSWorkspace.shared.activateFileViewerSelecting(urls)
    }

    /// The path with the home folder shown as ~.
    static func displayPath(_ path: String) -> String {
        (path as NSString).abbreviatingWithTildeInPath
    }
}

/// Asks for folders with the standard open panel, attached to the Settings
/// window.
@MainActor
enum FolderPicker {
    static func choose(
        message: String, startingAt directory: URL?, allowsMultipleSelection: Bool
    ) async -> [URL] {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = allowsMultipleSelection
        panel.message = message
        panel.prompt = "Choose"
        panel.directoryURL = directory
        let response: NSApplication.ModalResponse
        if let window = SettingsWindowController.settingsWindow ?? NSApp.keyWindow {
            response = await panel.beginSheetModal(for: window)
        } else {
            NSApp.activate()
            response = panel.runModal()
        }
        return response == .OK ? panel.urls : []
    }
}
