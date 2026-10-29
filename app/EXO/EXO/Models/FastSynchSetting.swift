import Foundation

/// Whether MLX synchronises the CPU and GPU by spinning on shared memory
/// ("fast synch"), passed to exo as EXO_FAST_SYNCH.
enum FastSynchSetting: String, CaseIterable, Identifiable {
    /// exo decides for each model: on for models that run over RDMA, where
    /// it lowers latency, and off otherwise.
    case automatic
    /// On for every model.
    case on
    /// Off for every model.
    case off

    static let storageKey = "EXOFastSynch"
    /// The on/off toggle earlier versions stored.
    static let legacyEnabledKey = "EXOFastSynchEnabled"

    var id: Self { self }

    var title: String {
        switch self {
        case .automatic: return "Automatic"
        case .on: return "On"
        case .off: return "Off"
        }
    }

    /// The value of EXO_FAST_SYNCH, or nil to leave the choice to exo.
    var environmentValue: String? {
        switch self {
        case .automatic: return nil
        case .on: return "true"
        case .off: return "false"
        }
    }

    /// Reads the setting, converting the toggle that earlier versions stored.
    ///
    /// That toggle defaulted to on and forced fast synch for every model, so
    /// a stored `true` is almost always the old default rather than a choice:
    /// it becomes Automatic. A stored `false` was chosen, and stays Off.
    static func load(from defaults: UserDefaults) -> FastSynchSetting {
        if let stored = defaults.string(forKey: storageKey),
            let setting = FastSynchSetting(rawValue: stored)
        {
            return setting
        }
        let legacy = defaults.object(forKey: legacyEnabledKey) as? Bool
        let setting: FastSynchSetting = legacy == false ? .off : .automatic
        defaults.set(setting.rawValue, forKey: storageKey)
        defaults.removeObject(forKey: legacyEnabledKey)
        return setting
    }
}
