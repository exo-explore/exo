import Foundation
import Testing

@testable import EXO

private func start(_ command: String) throws -> Process {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/sh")
    process.arguments = ["-c", command]
    try process.run()
    return process
}

struct ExoProcessStopTests {

    @Test func waitingForExitReturnsOnceTheProcessHasExited() async throws {
        let process = try start("sleep 0.3")
        let started = ContinuousClock.now

        let exited = await ExoProcessController.waitForExit(of: process, seconds: 5)

        #expect(exited)
        #expect(!process.isRunning)
        #expect(ContinuousClock.now - started < .seconds(3))
    }

    @Test func waitingForExitGivesUpAtTheDeadline() async throws {
        let process = try start("exec sleep 30")
        defer { process.terminate() }
        let started = ContinuousClock.now

        let exited = await ExoProcessController.waitForExit(of: process, seconds: 1)

        #expect(!exited)
        #expect(process.isRunning)
        let waited = ContinuousClock.now - started
        #expect(waited >= .seconds(1) && waited < .seconds(5))
    }
}
