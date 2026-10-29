import Foundation

/// How the menu bar icon shows what EXO is doing. The icon is a template
/// image, so the system colours it for the menu bar; state is shown by
/// opacity, motion and small badges, never by colour alone.
struct MenuBarIconStyle: Equatable {
    enum Opacity: Equatable {
        /// Running.
        case full
        /// Stopped or failed: about 35 %, like a disabled control.
        case dimmed
        /// Starting: fades in and out, or holds halfway with Reduce Motion on.
        case pulsing
    }

    /// Something needs the user: EXO failed or isn't responding, or a
    /// problem shows in the menu (Local Network access, a Thunderbolt Bridge
    /// loop).
    let needsAttention: Bool
    /// Sparkle found an update that hasn't been installed.
    let updateAvailable: Bool
    let opacity: Opacity
    /// Says the same in words, for VoiceOver and the tooltip.
    let accessibilityLabel: String

    static let dimmedAlpha = 0.35
    static let reducedMotionStartingAlpha = 0.6

    static func showing(
        _ status: ExoProcessController.Status, apiState: APIResponsiveness.State,
        problemInMenu: Bool, updateAvailable: Bool
    ) -> MenuBarIconStyle {
        let opacity: Opacity
        let state: String
        /// The state itself is the problem.
        let stateIsAProblem: Bool
        switch status {
        case .stopped:
            (opacity, state, stateIsAProblem) = (.dimmed, "stopped", false)
        case .starting:
            (opacity, state, stateIsAProblem) = (.pulsing, "starting", false)
        case .failed:
            (opacity, state, stateIsAProblem) = (.dimmed, "couldn't start", true)
        case .running where apiState == .notResponding:
            (opacity, state, stateIsAProblem) = (.full, "not responding", true)
        case .running:
            (opacity, state, stateIsAProblem) = (.full, "running", false)
        }
        // Problems shown in the menu only matter while EXO is on.
        let menuProblem = problemInMenu && status != .stopped
        let needsAttention = stateIsAProblem || menuProblem
        var label = "EXO, \(state)"
        if menuProblem && !stateIsAProblem {
            label += ", needs attention"
        }
        if updateAvailable {
            label += ", update available"
        }
        return MenuBarIconStyle(
            needsAttention: needsAttention, updateAvailable: updateAvailable, opacity: opacity,
            accessibilityLabel: label)
    }

    /// The icon's alpha at `phase` (0...1, one pulse) while starting.
    func alpha(atPulsePhase phase: Double, reduceMotion: Bool) -> Double {
        switch opacity {
        case .full:
            return 1
        case .dimmed:
            return Self.dimmedAlpha
        case .pulsing:
            if reduceMotion { return Self.reducedMotionStartingAlpha }
            // A cosine between the dimmed and the full icon.
            let wave = (1 - cos(phase * 2 * .pi)) / 2
            return Self.dimmedAlpha + (1 - Self.dimmedAlpha) * wave
        }
    }
}
