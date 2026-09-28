import SwiftUI

extension Binding where Value == [CustomEnvironmentVariable] {
    /// A binding to one field of the variable with `id`, found by id on every
    /// read and write.
    ///
    /// `ForEach($variables)` gives each row a binding to a fixed array index,
    /// so any read or write through it after the row has been removed goes
    /// past the end of the array (a crash) or to whichever row moved into the
    /// slot. Once the variable is gone, this binding reads as empty and
    /// ignores writes instead.
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
