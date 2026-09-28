import Foundation
import SwiftUI
import Testing

@testable import EXO

/// Holds the list the way `@State` does for the Settings view, so tests can
/// drive the same bindings the view uses.
private final class VariableListStore {
    var variables: [CustomEnvironmentVariable]

    init(_ variables: [CustomEnvironmentVariable]) {
        self.variables = variables
    }

    var binding: Binding<[CustomEnvironmentVariable]> {
        Binding(get: { self.variables }, set: { self.variables = $0 })
    }
}

struct CustomEnvironmentVariableTests {

    // MARK: - Removing a row while its fields are bound (#2113)

    @Test func removingANewBlankRowThenCommittingItsFieldDoesNotCrash() {
        // "Add Variable", leave it blank, press "-": the row's text field
        // commits its (empty) text as the row goes away.
        let blank = CustomEnvironmentVariable()
        let store = VariableListStore([blank])
        let keyField = store.binding.field(\.key, of: blank.id)
        let valueField = store.binding.field(\.value, of: blank.id)

        store.variables.removeAll { $0.id == blank.id }
        keyField.wrappedValue = ""
        valueField.wrappedValue = ""

        #expect(store.variables.isEmpty)
        #expect(keyField.wrappedValue == "")
        #expect(valueField.wrappedValue == "")
    }

    @Test func committingARemovedRowDoesNotOverwriteTheRowThatTookItsPlace() {
        let first = CustomEnvironmentVariable(key: "FIRST", value: "1")
        let removed = CustomEnvironmentVariable()
        let last = CustomEnvironmentVariable(key: "LAST", value: "3")
        let store = VariableListStore([first, removed, last])
        let removedKeyField = store.binding.field(\.key, of: removed.id)
        let removedValueField = store.binding.field(\.value, of: removed.id)

        store.variables.removeAll { $0.id == removed.id }
        removedKeyField.wrappedValue = ""
        removedValueField.wrappedValue = ""

        #expect(store.variables == [first, last])
    }

    @Test func fieldBindingEditsOnlyItsOwnVariable() {
        let first = CustomEnvironmentVariable(key: "FIRST", value: "1")
        let second = CustomEnvironmentVariable(key: "SECOND", value: "2")
        let store = VariableListStore([first, second])

        store.binding.field(\.key, of: second.id).wrappedValue = "RENAMED"
        store.binding.field(\.value, of: second.id).wrappedValue = "two"

        #expect(store.variables[0] == first)
        #expect(store.variables[1].key == "RENAMED")
        #expect(store.variables[1].value == "two")
        #expect(store.binding.field(\.key, of: second.id).wrappedValue == "RENAMED")
    }

    // MARK: - Only valid entries are saved or passed to exo

    @Test(arguments: ["FOO", "foo", "_", "_FOO_1", "EXO_OFFLINE", "HF_TOKEN", "a1"])
    func acceptsValidNames(_ name: String) {
        #expect(CustomEnvironmentVariable.isValidName(name))
    }

    @Test(arguments: ["", " ", "1FOO", "FOO BAR", "FOO=BAR", "FOO-BAR", "FOO\n", "ñ", "ÀB"])
    func rejectsInvalidNames(_ name: String) {
        #expect(!CustomEnvironmentVariable.isValidName(name))
    }

    @Test func sanitizedDropsBlankAndInvalidEntriesAndTrimsKeys() {
        let variables = [
            CustomEnvironmentVariable(key: "", value: ""),
            CustomEnvironmentVariable(key: "   ", value: "ignored"),
            CustomEnvironmentVariable(key: "1FOO", value: "x"),
            CustomEnvironmentVariable(key: "FOO=BAR", value: "x"),
            CustomEnvironmentVariable(key: "  PADDED  ", value: " value kept as is "),
            CustomEnvironmentVariable(key: "EMPTY_VALUE", value: ""),
        ]

        let sanitized = CustomEnvironmentVariable.sanitized(variables)

        #expect(sanitized.map(\.key) == ["PADDED", "EMPTY_VALUE"])
        #expect(sanitized.map(\.value) == [" value kept as is ", ""])
        #expect(sanitized.map(\.id) == [variables[4].id, variables[5].id])
    }

    @Test func sanitizedTrimsNewlinesFromPastedKeys() {
        let variables = [CustomEnvironmentVariable(key: "PASTED\n", value: "x")]

        #expect(CustomEnvironmentVariable.sanitized(variables).map(\.key) == ["PASTED"])
    }

    @Test func onlyFilledInInvalidNamesAreFlagged() {
        let flagged = ["1FOO", "FOO BAR", "A=B", " FOO-BAR\n"]
        let notFlagged = ["", "   ", "\n", "FOO", " FOO ", "FOO\n"]

        for key in flagged {
            #expect(CustomEnvironmentVariable(key: key).hasInvalidName, "\(key.debugDescription)")
        }
        for key in notFlagged {
            #expect(!CustomEnvironmentVariable(key: key).hasInvalidName, "\(key.debugDescription)")
        }
    }

    @Test func sanitizedKeepsTheLastOccurrenceOfADuplicateKey() {
        let variables = [
            CustomEnvironmentVariable(key: "A", value: "first"),
            CustomEnvironmentVariable(key: "B", value: "b"),
            CustomEnvironmentVariable(key: " A", value: "second"),
        ]

        let sanitized = CustomEnvironmentVariable.sanitized(variables)

        #expect(sanitized.map(\.key) == ["B", "A"])
        #expect(sanitized.map(\.value) == ["b", "second"])
    }
}
