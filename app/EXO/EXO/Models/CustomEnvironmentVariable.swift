import Foundation

/// A user-defined environment variable that is injected into the exo child
/// process at launch. Used as an escape hatch for env vars that don't have
/// first-class typed UI in Settings.
struct CustomEnvironmentVariable: Codable, Identifiable, Equatable {
    var id: UUID
    var key: String
    var value: String

    init(id: UUID = UUID(), key: String = "", value: String = "") {
        self.id = id
        self.key = key
        self.value = value
    }

    private static let nameHeadCharacters = CharacterSet(
        charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
    )
    private static let nameTailCharacters = nameHeadCharacters.union(
        CharacterSet(charactersIn: "0123456789")
    )

    /// Whether `name` is a POSIX-style environment variable name,
    /// `[A-Za-z_][A-Za-z0-9_]*`. ASCII only, so Unicode letters (e.g. `ñ`,
    /// Cyrillic) are rejected. The empty string is not a valid name.
    static func isValidName(_ name: String) -> Bool {
        guard let first = name.unicodeScalars.first, nameHeadCharacters.contains(first) else {
            return false
        }
        return name.unicodeScalars.dropFirst().allSatisfy { nameTailCharacters.contains($0) }
    }

    /// The key as it is saved and passed to exo: without surrounding
    /// whitespace or newlines (e.g. from a paste).
    var trimmedKey: String {
        key.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Whether the key is filled in but isn't a valid name. Blank rows don't
    /// count: they are simply dropped.
    var hasInvalidName: Bool {
        !trimmedKey.isEmpty && !Self.isValidName(trimmedKey)
    }

    /// The variables that may be stored or passed to exo: keys are trimmed,
    /// entries whose key is then empty or not a valid name are dropped, and
    /// duplicate keys keep only their last occurrence (the one that would win
    /// when the environment is built).
    static func sanitized(_ variables: [CustomEnvironmentVariable]) -> [CustomEnvironmentVariable] {
        let valid = variables.compactMap { variable -> CustomEnvironmentVariable? in
            let key = variable.trimmedKey
            guard isValidName(key) else { return nil }
            return CustomEnvironmentVariable(id: variable.id, key: key, value: variable.value)
        }
        var seenKeys = Set<String>()
        let lastOccurrences = valid.reversed().filter { seenKeys.insert($0.key).inserted }
        return Array(lastOccurrences.reversed())
    }
}
