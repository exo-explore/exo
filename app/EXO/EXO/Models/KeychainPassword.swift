import Foundation
import Security

/// A generic password in the user's login keychain.
struct KeychainPassword: Sendable {
    let service: String
    let account: String

    struct Failure: LocalizedError {
        let status: OSStatus

        var errorDescription: String? {
            let message = SecCopyErrorMessageString(status, nil) as String? ?? "error \(status)"
            return "The keychain couldn't be used: \(message)"
        }
    }

    private var query: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }

    /// The password, or nil if there is none.
    func read() throws -> String? {
        var query = self.query
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound {
            return nil
        }
        guard status == errSecSuccess, let data = result as? Data else {
            throw Failure(status: status)
        }
        return String(decoding: data, as: UTF8.self)
    }

    /// Saves `password`, replacing any already saved.
    func save(_ password: String) throws {
        let data = Data(password.utf8)
        let update = SecItemUpdate(
            query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if update == errSecSuccess {
            return
        }
        guard update == errSecItemNotFound else {
            throw Failure(status: update)
        }
        var item = query
        item[kSecValueData as String] = data
        item[kSecAttrLabel as String] = "\(service) (\(account))"
        let add = SecItemAdd(item as CFDictionary, nil)
        guard add == errSecSuccess else {
            throw Failure(status: add)
        }
    }

    /// Deletes the password. Deleting one that isn't there succeeds.
    func delete() throws {
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw Failure(status: status)
        }
    }
}

/// Where the Hugging Face access token is kept: in the keychain rather than
/// in EXO's preferences file, where earlier versions kept it in plain text.
enum HuggingFaceToken {
    static let keychain = KeychainPassword(service: "exolabs.EXO", account: "huggingface")
    /// Where earlier versions stored the token.
    static let legacyDefaultsKey = "EXOHFToken"

    /// Reads the token, first moving one stored by an earlier version from
    /// `defaults` into the keychain. The old copy is removed only once the
    /// keychain has it, so a keychain error never loses the token.
    static func load(from keychain: KeychainPassword, migrating defaults: UserDefaults)
        -> String
    {
        let stored = try? keychain.read()
        let legacy = defaults.string(forKey: legacyDefaultsKey) ?? ""
        if legacy.isEmpty {
            defaults.removeObject(forKey: legacyDefaultsKey)
            return stored ?? ""
        }
        if let stored, !stored.isEmpty {
            // Already moved; the keychain's copy is the current one.
            defaults.removeObject(forKey: legacyDefaultsKey)
            return stored
        }
        do {
            try keychain.save(legacy)
            defaults.removeObject(forKey: legacyDefaultsKey)
        } catch {
            // Kept where it was, and tried again at the next launch.
        }
        return legacy
    }

    /// Saves the token, or deletes it when it is empty.
    static func save(_ token: String, to keychain: KeychainPassword) throws {
        if token.isEmpty {
            try keychain.delete()
        } else {
            try keychain.save(token)
        }
    }
}
