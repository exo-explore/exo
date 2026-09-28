import Foundation
import SwiftUI

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

    /// The variables that may be stored or passed to exo: keys are trimmed of
    /// surrounding whitespace, entries whose key is then empty or not a valid
    /// name are dropped, and duplicate keys keep only their last occurrence
    /// (the one that would win when the environment is built).
    static func sanitized(_ variables: [CustomEnvironmentVariable]) -> [CustomEnvironmentVariable] {
        let valid = variables.compactMap { variable -> CustomEnvironmentVariable? in
            let key = variable.key.trimmingCharacters(in: .whitespaces)
            guard isValidName(key) else { return nil }
            return CustomEnvironmentVariable(id: variable.id, key: key, value: variable.value)
        }
        var seenKeys = Set<String>()
        let lastOccurrences = valid.reversed().filter { seenKeys.insert($0.key).inserted }
        return Array(lastOccurrences.reversed())
    }
}

extension Binding where Value == [CustomEnvironmentVariable] {
    /// A binding to one field of the variable with `id`, found by id on every
    /// read and write.
    ///
    /// `ForEach($variables)` gives each row a binding to a fixed array index.
    /// When a row is removed while one of its text fields is being edited, the
    /// field commits its text as it goes away and writes through that stale
    /// index: past the end of the array (a crash) or into whichever row moved
    /// into the slot. Once the variable is gone, this binding reads as empty
    /// and ignores writes instead.
    func field(
        _ keyPath: WritableKeyPath<CustomEnvironmentVariable, String>,
        of id: UUID
    ) -> Binding<String> {
        Binding<String>(
            get: {
                wrappedValue.first { $0.id == id }?[keyPath: keyPath] ?? ""
            },
            set: { newValue in
                guard let index = wrappedValue.firstIndex(where: { $0.id == id }) else {
                    return
                }
                wrappedValue[index][keyPath: keyPath] = newValue
            }
        )
    }
}
