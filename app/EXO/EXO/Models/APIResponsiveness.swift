import Foundation

/// Whether exo's API is answering the app's polls.
///
/// One slow or failed poll isn't enough to call exo unresponsive, and an exo
/// that hasn't answered yet since it was started is still connecting.
struct APIResponsiveness: Equatable {
    enum State: Equatable {
        /// No answer yet since exo was (re)started.
        case connecting
        case responding
        /// exo answered before, but the last few polls failed.
        case notResponding
    }

    /// Failed polls in a row, after exo has answered, that make it count as
    /// not responding.
    static let failuresBeforeNotResponding = 3

    private(set) var state: State = .connecting
    private var consecutiveFailures = 0

    mutating func recordAnswer() {
        state = .responding
        consecutiveFailures = 0
    }

    mutating func recordFailure() {
        consecutiveFailures += 1
        if state != .connecting && consecutiveFailures >= Self.failuresBeforeNotResponding {
            state = .notResponding
        }
    }
}
