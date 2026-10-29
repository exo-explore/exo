import Foundation

/// The first-run welcome window's three panes, as a state machine:
/// Welcome (with "Open EXO at login"), then Find Your Other Macs (which
/// explains the Local Network prompt before macOS shows it), then Ready.
struct WelcomeFlow: Equatable {
    enum Pane: Equatable {
        case welcome
        case findMacs
        case ready
    }

    /// Local Network access, as far as the welcome window knows.
    enum LocalNetwork: Equatable {
        /// Not asked yet: exo hasn't been started.
        case notAsked
        /// exo has been started, so macOS is asking (or has its answer).
        case asking
        case allowed
        /// Off for EXO; EXO still works on this Mac.
        case denied
    }

    enum Event: Equatable {
        /// Continue on the Welcome pane.
        case continueFromWelcome
        /// Continue on the Find Your Other Macs pane: starts exo, which makes
        /// macOS ask for Local Network access.
        case askForLocalNetwork
        case localNetworkAllowed
        case localNetworkDenied
        /// The check couldn't tell (e.g. it timed out); don't hold the user up.
        case localNetworkUndetermined
        /// Continue after a denial.
        case continueWithoutLocalNetwork
    }

    /// What the window does in response.
    enum Effect: Equatable {
        case startExo
    }

    private(set) var pane: Pane = .welcome
    private(set) var localNetwork: LocalNetwork = .notAsked

    /// Applies `event` and returns what to do.
    @discardableResult
    mutating func handle(_ event: Event) -> [Effect] {
        switch (pane, localNetwork, event) {
        case (.welcome, _, .continueFromWelcome):
            pane = .findMacs
        case (.findMacs, .notAsked, .askForLocalNetwork):
            localNetwork = .asking
            return [.startExo]
        case (.findMacs, .asking, .localNetworkAllowed), (.findMacs, .denied, .localNetworkAllowed):
            localNetwork = .allowed
            pane = .ready
        case (.findMacs, .asking, .localNetworkDenied):
            localNetwork = .denied
        case (.findMacs, .asking, .localNetworkUndetermined):
            pane = .ready
        case (.findMacs, .denied, .continueWithoutLocalNetwork):
            pane = .ready
        case (.ready, _, .localNetworkAllowed):
            localNetwork = .allowed
        default:
            break
        }
        return []
    }
}
