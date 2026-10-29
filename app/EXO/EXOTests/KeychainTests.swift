import Foundation
import Testing

@testable import EXO

/// A keychain item under a service name of its own, deleted afterwards, so
/// tests never touch EXO's real token.
final class ScratchKeychainItem {
    let password = KeychainPassword(
        service: "exolabs.EXO.tests.\(UUID().uuidString)", account: "huggingface")

    deinit {
        try? password.delete()
    }
}

struct KeychainTests {

    @Test func savesReadsReplacesAndDeletes() throws {
        let item = ScratchKeychainItem()

        #expect(try item.password.read() == nil)
        try item.password.save("hf_first")
        #expect(try item.password.read() == "hf_first")
        try item.password.save("hf_second")
        #expect(try item.password.read() == "hf_second")
        try item.password.delete()
        #expect(try item.password.read() == nil)
        // Deleting what isn't there is fine.
        try item.password.delete()
    }

    @Test func aTokenStoredByAnEarlierVersionMovesToTheKeychain() throws {
        let item = ScratchKeychainItem()
        let storage = ScratchDefaults()
        storage.defaults.set("hf_legacy", forKey: HuggingFaceToken.legacyDefaultsKey)

        let token = HuggingFaceToken.load(from: item.password, migrating: storage.defaults)

        #expect(token == "hf_legacy")
        #expect(try item.password.read() == "hf_legacy")
        // No plain-text copy is left in the preferences.
        #expect(storage.defaults.object(forKey: HuggingFaceToken.legacyDefaultsKey) == nil)
        // The next launch reads it from the keychain.
        #expect(
            HuggingFaceToken.load(from: item.password, migrating: storage.defaults) == "hf_legacy")
    }

    @Test func theKeychainCopyWinsOverALeftoverPlainTextOne() throws {
        let item = ScratchKeychainItem()
        let storage = ScratchDefaults()
        try item.password.save("hf_current")
        storage.defaults.set("hf_stale", forKey: HuggingFaceToken.legacyDefaultsKey)

        #expect(
            HuggingFaceToken.load(from: item.password, migrating: storage.defaults) == "hf_current")
        #expect(storage.defaults.object(forKey: HuggingFaceToken.legacyDefaultsKey) == nil)
    }

    @Test func anEmptyOldTokenIsDroppedAndNothingIsSaved() throws {
        let item = ScratchKeychainItem()
        let storage = ScratchDefaults()
        storage.defaults.set("", forKey: HuggingFaceToken.legacyDefaultsKey)

        #expect(HuggingFaceToken.load(from: item.password, migrating: storage.defaults) == "")
        #expect(try item.password.read() == nil)
        #expect(storage.defaults.object(forKey: HuggingFaceToken.legacyDefaultsKey) == nil)
    }

    @Test func clearingTheTokenDeletesIt() throws {
        let item = ScratchKeychainItem()
        try HuggingFaceToken.save("hf_x", to: item.password)
        #expect(try item.password.read() == "hf_x")
        try HuggingFaceToken.save("", to: item.password)
        #expect(try item.password.read() == nil)
    }

    @MainActor
    @Test func theSettingIsKeptInTheKeychainNotThePreferences() throws {
        let item = ScratchKeychainItem()
        let storage = ScratchDefaults()
        let controller = ExoProcessController(
            readinessProbe: { false }, defaults: storage.defaults, tokenKeychain: item.password)

        controller.hfToken = "hf_typed"

        #expect(try item.password.read() == "hf_typed")
        #expect(storage.defaults.object(forKey: HuggingFaceToken.legacyDefaultsKey) == nil)
        #expect(controller.launchSettings.environment(buildTag: "t")["HF_TOKEN"] == "hf_typed")
        let reloaded = ExoProcessController(
            readinessProbe: { false }, defaults: storage.defaults, tokenKeychain: item.password)
        #expect(reloaded.hfToken == "hf_typed")
    }
}
