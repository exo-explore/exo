import Foundation

/// What to do once settings that exo reads at startup have changed.
enum SettingsRestartPolicy {
    enum Decision: Equatable {
        /// exo already has these settings, or isn't running and gets them
        /// when it next starts.
        case nothing
        /// Restart exo now: nothing running on this Mac would stop.
        case restartNow
        /// Ask first, because restarting would stop models running on this
        /// Mac, or it can't be told whether it would.
        case ask
    }

    /// - Parameters:
    ///   - status: exo's status.
    ///   - runningEnvironment: the settings environment exo was started
    ///     with, or nil if it isn't running.
    ///   - settingsEnvironment: the settings environment it would get now.
    ///   - modelsRunningOnThisMac: whether a model has a part running on this
    ///     Mac, or nil if that isn't known.
    static func decide(
        status: ExoProcessController.Status,
        runningEnvironment: [String: String]?,
        settingsEnvironment: [String: String],
        modelsRunningOnThisMac: Bool?
    ) -> Decision {
        guard let runningEnvironment, runningEnvironment != settingsEnvironment else {
            return .nothing
        }
        switch status {
        case .stopped, .failed:
            return .nothing
        case .starting:
            // Nothing has been placed on an exo that is still starting.
            return .restartNow
        case .running:
            return modelsRunningOnThisMac == false ? .restartNow : .ask
        }
    }
}
