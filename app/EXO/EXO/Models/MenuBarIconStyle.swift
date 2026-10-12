import AppKit

/// How the menu bar icon shows what exo is doing, so that a stopped, failed
/// or unresponsive exo doesn't look like a healthy one.
struct MenuBarIconStyle: Equatable {
    enum Badge: Equatable {
        /// exo is running but has stopped answering.
        case warning
        /// exo exited on its own, or couldn't be started.
        case error

        var color: NSColor {
            switch self {
            case .warning:
                return .systemOrange
            case .error:
                return .systemRed
            }
        }
    }

    /// Drawn in grey when exo isn't running.
    let isGreyscale: Bool
    /// A dot in the corner of the icon, for a problem that needs attention.
    let badge: Badge?
    /// Says what the badge and the grey show, for VoiceOver.
    let accessibilityLabel: String

    static func showing(
        _ status: ExoProcessController.Status, apiState: APIResponsiveness.State
    ) -> MenuBarIconStyle {
        switch status {
        case .stopped:
            return MenuBarIconStyle(
                isGreyscale: true, badge: nil, accessibilityLabel: "EXO, stopped")
        case .failed:
            return MenuBarIconStyle(
                isGreyscale: true, badge: .error, accessibilityLabel: "EXO, failed")
        case .starting:
            return MenuBarIconStyle(
                isGreyscale: false, badge: nil, accessibilityLabel: "EXO, starting")
        case .running:
            if apiState == .notResponding {
                return MenuBarIconStyle(
                    isGreyscale: false, badge: .warning,
                    accessibilityLabel: "EXO, not responding")
            }
            return MenuBarIconStyle(isGreyscale: false, badge: nil, accessibilityLabel: "EXO")
        }
    }
}
