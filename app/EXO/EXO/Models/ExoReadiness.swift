import Foundation

/// Whether a freshly started exo is up: until its API answers, the app shows
/// "Starting…" rather than "Running".
enum ExoReadiness {
    /// Answers once exo's HTTP API is serving; it doesn't depend on the
    /// cluster or an election.
    static let probeURL = URL(string: "http://127.0.0.1:52415/node_id")!

    /// Calls `probe` every `interval` until it returns true. Returns false if
    /// `isAlive` turns false (exo exited) or the task is cancelled first.
    static func wait(
        every interval: Duration, while isAlive: () -> Bool, until probe: () async -> Bool
    ) async -> Bool {
        while !Task.isCancelled && isAlive() {
            if await probe() {
                return true
            }
            try? await Task.sleep(for: interval)
        }
        return false
    }

    /// No on-disk cache, like `ClusterStateService`'s session.
    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        return URLSession(configuration: configuration)
    }()

    /// Whether a GET of `url` succeeds.
    static func answers(_ url: URL = probeURL, session: URLSession = session) async -> Bool {
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 2
        guard let (_, response) = try? await session.data(for: request),
            let httpResponse = response as? HTTPURLResponse
        else {
            return false
        }
        return (200..<300).contains(httpResponse.statusCode)
    }
}
