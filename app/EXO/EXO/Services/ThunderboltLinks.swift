import Foundation
import SystemConfiguration

/// Whether this Mac has a Thunderbolt cable to another computer: then EXO's
/// Thunderbolt networking is worth offering. Read in-process from
/// SystemConfiguration, cheap enough to check when the menu opens.
enum ThunderboltLinks {
    /// The BSD names of the Mac's Thunderbolt ports (en2, en3, ...), not
    /// counting Thunderbolt Bridge.
    static func portNames() -> [String] {
        guard let interfaces = SCNetworkInterfaceCopyAll() as? [SCNetworkInterface] else {
            return []
        }
        return interfaces.compactMap { interface in
            guard let name = SCNetworkInterfaceGetBSDName(interface) as String?,
                let displayName = SCNetworkInterfaceGetLocalizedDisplayName(interface) as String?
            else { return nil }
            return isThunderboltPort(bsdName: name, displayName: displayName) ? name : nil
        }
    }

    static func isThunderboltPort(bsdName: String, displayName: String) -> Bool {
        displayName.hasPrefix("Thunderbolt") && !bsdName.hasPrefix("bridge")
    }

    /// Whether any Thunderbolt port has a live link.
    static func anyLinkActive() -> Bool {
        guard let store = SCDynamicStoreCreate(nil, "EXO" as CFString, nil, nil) else {
            return false
        }
        return portNames().contains { name in
            let key = "State:/Network/Interface/\(name)/Link" as CFString
            let link = SCDynamicStoreCopyValue(store, key) as? [String: Any]
            return link?["Active"] as? Bool ?? false
        }
    }
}
