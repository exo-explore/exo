import Foundation

/// A folder, besides the download folder, where exo looks for models.
struct ModelFolder: Codable, Equatable, Identifiable {
    var id: UUID
    var path: String
    /// exo never downloads into, or deletes from, a read-only folder.
    var readOnly: Bool

    init(id: UUID = UUID(), path: String, readOnly: Bool = false) {
        self.id = id
        self.path = path
        self.readOnly = readOnly
    }

    static let storageKey = "EXOOtherModelFolders"
    /// The colon-separated lists earlier versions stored.
    static let legacyWritableKey = "EXOAdditionalModelsDirs"
    static let legacyReadOnlyKey = "EXOReadOnlyModelsDirs"

    var trimmedPath: String {
        path.trimmingCharacters(in: .whitespaces)
    }

    /// exo takes the folders as colon-separated lists, so a folder whose path
    /// contains a colon can't be passed to it.
    var isUsable: Bool {
        !trimmedPath.isEmpty && !trimmedPath.contains(":")
    }

    /// The folders as `EXO_MODELS_DIRS` and `EXO_MODELS_READ_ONLY_DIRS` take
    /// them, leaving out the ones that can't be passed.
    static func environment(for folders: [ModelFolder]) -> [String: String] {
        let usable = folders.filter(\.isUsable)
        var environment: [String: String] = [:]
        let writable = usable.filter { !$0.readOnly }.map(\.trimmedPath)
        if !writable.isEmpty {
            environment["EXO_MODELS_DIRS"] = writable.joined(separator: ":")
        }
        let readOnly = usable.filter(\.readOnly).map(\.trimmedPath)
        if !readOnly.isEmpty {
            environment["EXO_MODELS_READ_ONLY_DIRS"] = readOnly.joined(separator: ":")
        }
        return environment
    }

    /// Reads the folders, converting the two text fields that earlier versions
    /// stored. exo treats a folder that is in both lists as read-only, so it
    /// becomes one read-only folder.
    static func load(from defaults: UserDefaults) -> [ModelFolder] {
        if let data = defaults.data(forKey: storageKey),
            let folders = try? JSONDecoder().decode([ModelFolder].self, from: data)
        {
            return folders
        }
        let readOnly = paths(in: defaults.string(forKey: legacyReadOnlyKey))
        let writable = paths(in: defaults.string(forKey: legacyWritableKey))
            .filter { !readOnly.contains($0) }
        let folders =
            writable.map { ModelFolder(path: $0) }
            + readOnly.map { ModelFolder(path: $0, readOnly: true) }
        save(folders, to: defaults)
        defaults.removeObject(forKey: legacyWritableKey)
        defaults.removeObject(forKey: legacyReadOnlyKey)
        return folders
    }

    static func save(_ folders: [ModelFolder], to defaults: UserDefaults) {
        guard let data = try? JSONEncoder().encode(folders) else { return }
        defaults.set(data, forKey: storageKey)
    }

    private static func paths(in list: String?) -> [String] {
        (list ?? "")
            .split(separator: ":")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
}
