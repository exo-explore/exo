import Foundation
import Testing

@testable import EXO

/// Answers every request with a fixed HTTP status, or fails to connect.
private final class FixedStatusProtocol: URLProtocol {
    private static let lock = NSLock()
    private static var statusByHost: [String: Int] = [:]

    static func answer(_ status: Int?, for host: String) {
        lock.lock()
        defer { lock.unlock() }
        statusByHost[host] = status
    }

    private static func status(for host: String) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        return statusByHost[host]
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let status = Self.status(for: url.host ?? "") else {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
            return
        }
        let response = HTTPURLResponse(
            url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(#""node-a""#.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Counts probes, answering "up" from the `upFrom`-th one.
private final class Probe: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    let upFrom: Int

    init(upFrom: Int) {
        self.upFrom = upFrom
    }

    var calls: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func check() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        count += 1
        return count >= upFrom
    }
}

struct ExoReadinessTests {

    @Test func waitsUntilTheAPIAnswers() async {
        let probe = Probe(upFrom: 3)
        let ready = await ExoReadiness.wait(
            every: .milliseconds(1), while: { true }, until: { probe.check() })
        #expect(ready)
        #expect(probe.calls == 3)
    }

    @Test func givesUpWhenExoExitsBeforeItIsUp() async {
        let probe = Probe(upFrom: .max)
        let ready = await ExoReadiness.wait(
            every: .milliseconds(1), while: { probe.calls < 2 }, until: { probe.check() })
        #expect(!ready)
        #expect(probe.calls == 2)
    }

    @Test func givesUpWhenCancelled() async {
        let probe = Probe(upFrom: .max)
        let waiting = Task {
            await ExoReadiness.wait(
                every: .milliseconds(5), while: { true }, until: { probe.check() })
        }
        try? await Task.sleep(for: .milliseconds(30))
        waiting.cancel()
        #expect(await waiting.value == false)
    }

    @Test func onlyASuccessfulAnswerCountsAsUp() async {
        let host = "exo-\(UUID().uuidString.lowercased()).test"
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FixedStatusProtocol.self]
        let session = URLSession(configuration: configuration)
        let url = URL(string: "http://\(host):52415/node_id")!

        #expect(await ExoReadiness.answers(url, session: session) == false)
        FixedStatusProtocol.answer(503, for: host)
        #expect(await ExoReadiness.answers(url, session: session) == false)
        FixedStatusProtocol.answer(200, for: host)
        #expect(await ExoReadiness.answers(url, session: session))
    }

    @Test func probesExosNodeIDEndpoint() {
        #expect(ExoReadiness.probeURL.absoluteString == "http://127.0.0.1:52415/node_id")
    }
}
