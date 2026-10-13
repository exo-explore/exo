import Combine
import Foundation
import Testing

@testable import EXO

/// Serves canned answers to `ClusterStateService` instead of a real exo. Each
/// test registers a handler under its own host name, so tests running in
/// parallel don't see each other's answers.
private final class StubExoAPI: URLProtocol {
    enum Answer {
        case state
        case unreachable
    }

    private static let lock = NSLock()
    private static var answers: [String: Answer] = [:]

    static func set(_ answer: Answer, for host: String) {
        lock.lock()
        defer { lock.unlock() }
        answers[host] = answer
    }

    private static func answer(for host: String) -> Answer {
        lock.lock()
        defer { lock.unlock() }
        return answers[host] ?? .unreachable
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let host = url.host else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        switch Self.answer(for: host) {
        case .unreachable:
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
        case .state:
            let body: String
            switch url.lastPathComponent {
            case "state":
                body = #"{"instances": {}, "runners": {}}"#
            case "node_id":
                body = #""node-a""#
            default:
                body = #"{"data": []}"#
            }
            let response = HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(body.utf8))
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() {}
}

@MainActor
private func makeService() -> (ClusterStateService, host: String) {
    let host = "exo-\(UUID().uuidString.lowercased()).test"
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [StubExoAPI.self]
    let service = ClusterStateService(
        baseURL: URL(string: "http://\(host):52415")!,
        session: URLSession(configuration: configuration)
    )
    return (service, host)
}

struct APIResponsivenessTests {

    @Test func exoThatHasNotAnsweredYetIsStillConnecting() {
        var responsiveness = APIResponsiveness()
        for _ in 0..<20 {
            responsiveness.recordFailure()
        }
        #expect(responsiveness.state == .connecting)
    }

    @Test func aFewFailedPollsInARowMakeExoNotResponding() {
        var responsiveness = APIResponsiveness()
        responsiveness.recordAnswer()
        for _ in 1..<APIResponsiveness.failuresBeforeNotResponding {
            responsiveness.recordFailure()
            #expect(responsiveness.state == .responding)
        }
        responsiveness.recordFailure()
        #expect(responsiveness.state == .notResponding)

        responsiveness.recordAnswer()
        #expect(responsiveness.state == .responding)
    }

    @Test func anAnswerBetweenFailuresStartsTheCountAgain() {
        var responsiveness = APIResponsiveness()
        responsiveness.recordAnswer()
        for _ in 0..<5 {
            for _ in 1..<APIResponsiveness.failuresBeforeNotResponding {
                responsiveness.recordFailure()
            }
            responsiveness.recordAnswer()
        }
        #expect(responsiveness.state == .responding)
    }
}

struct MenuBarIconStyleTests {

    @Test func onlyAHealthyOrStartingExoLooksHealthy() {
        let healthy = MenuBarIconStyle.showing(.running, apiState: .responding)
        #expect(!healthy.isGreyscale)
        #expect(healthy.badge == nil)

        let starting = MenuBarIconStyle.showing(.starting, apiState: .connecting)
        #expect(!starting.isGreyscale)
        #expect(starting.badge == nil)
    }

    @Test func failedExoIsGreyWithARedBadge() {
        for apiState in [APIResponsiveness.State.connecting, .responding, .notResponding] {
            let style = MenuBarIconStyle.showing(
                .failed(message: "Exited with code 1"), apiState: apiState)
            #expect(style.isGreyscale)
            #expect(style.badge == .error)
            #expect(style.badge?.color == .systemRed)
            #expect(style.accessibilityLabel == "EXO, failed")
        }
    }

    @Test func unresponsiveExoGetsAWarningBadge() {
        let style = MenuBarIconStyle.showing(.running, apiState: .notResponding)
        #expect(!style.isGreyscale)
        #expect(style.badge == .warning)
        #expect(style.accessibilityLabel == "EXO, not responding")
    }

    @Test func stoppedExoIsGreyWithoutABadge() {
        let style = MenuBarIconStyle.showing(.stopped, apiState: .notResponding)
        #expect(style.isGreyscale)
        #expect(style.badge == nil)
    }
}

@MainActor
struct ClusterStateServiceHealthTests {

    @Test func statsAreClearedOnceExoStopsAnswering() async {
        let (service, host) = makeService()
        StubExoAPI.set(.state, for: host)
        await service.fetchSnapshot()
        #expect(service.latestSnapshot != nil)
        #expect(service.apiState == .responding)

        StubExoAPI.set(.unreachable, for: host)
        for _ in 1..<APIResponsiveness.failuresBeforeNotResponding {
            await service.fetchSnapshot()
        }
        // A blip keeps the stats on screen.
        #expect(service.latestSnapshot != nil)
        #expect(service.apiState == .responding)

        await service.fetchSnapshot()
        #expect(service.latestSnapshot == nil)
        #expect(service.apiState == .notResponding)

        StubExoAPI.set(.state, for: host)
        await service.fetchSnapshot()
        #expect(service.latestSnapshot != nil)
        #expect(service.apiState == .responding)
    }

    @Test func startingExoAgainForgetsThePreviousRun() async {
        let (service, host) = makeService()
        let statuses = CurrentValueSubject<ExoProcessController.Status, Never>(.running)
        service.resetWhenExoStarts(statuses)
        StubExoAPI.set(.state, for: host)
        await service.fetchSnapshot()
        #expect(service.localNodeId == "node-a")

        // Save & Restart: the old exo goes away and a new one starts.
        StubExoAPI.set(.unreachable, for: host)
        statuses.send(.stopped)
        statuses.send(.starting)
        #expect(service.latestSnapshot == nil)
        #expect(service.localNodeId == nil)
        #expect(service.apiState == .connecting)

        // The new exo taking a while to answer isn't "not responding".
        for _ in 0..<(APIResponsiveness.failuresBeforeNotResponding * 2) {
            await service.fetchSnapshot()
        }
        #expect(service.apiState == .connecting)
    }
}
