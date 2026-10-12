import Foundation
import Testing

@testable import EXO

/// A log file in its own temporary directory, removed afterwards.
private final class TemporaryLog {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("EXOTests-log-\(UUID().uuidString)", isDirectory: true)
    let log: ExoLogFile

    init(maxBytes: UInt64 = ExoLogFile.defaultMaxBytes) {
        log = ExoLogFile(
            url: directory.appendingPathComponent("exo_app.log"), maxBytes: maxBytes)
    }

    func contents(of url: URL) -> String {
        (try? String(contentsOf: url, encoding: .utf8)) ?? ""
    }

    deinit {
        try? FileManager.default.removeItem(at: directory)
    }
}

/// Runs a shell command the way the app runs exo: stdout and stderr both
/// appended to the log.
private func runWithOutput(
    to log: ExoLogFile, _ command: String
) throws -> (Process.TerminationReason, Int32) {
    let handle = try log.openForAppending()
    defer { try? handle.close() }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/sh")
    process.arguments = ["-c", command]
    process.standardOutput = handle
    process.standardError = handle
    try process.run()
    process.waitUntilExit()
    return (process.terminationReason, process.terminationStatus)
}

struct ExoLogFileTests {

    @Test func capturesStdoutAndStderrAndAppendsAcrossRuns() throws {
        let temporary = TemporaryLog()
        _ = try runWithOutput(to: temporary.log, "echo first; echo oops >&2")
        _ = try runWithOutput(to: temporary.log, "echo second")

        #expect(temporary.contents(of: temporary.log.url) == "first\noops\nsecond\n")
    }

    @Test func staysUnderItsLimitByRotatingIntoASingleOlderCopy() throws {
        let temporary = TemporaryLog(maxBytes: 10)
        let handle = try temporary.log.openForAppending()
        defer { try? handle.close() }

        try handle.write(contentsOf: Data("0123456789".utf8))
        #expect(try !temporary.log.rotateIfNeeded())

        try handle.write(contentsOf: Data("ABC".utf8))
        #expect(try temporary.log.rotateIfNeeded())
        #expect(temporary.contents(of: temporary.log.rotatedURL) == "0123456789ABC")
        #expect(temporary.log.size() == 0)

        try handle.write(contentsOf: Data("second batch".utf8))
        #expect(try temporary.log.rotateIfNeeded())
        #expect(temporary.contents(of: temporary.log.rotatedURL) == "second batch")
        let files = try FileManager.default.contentsOfDirectory(atPath: temporary.directory.path)
        #expect(files.sorted() == ["exo_app.log", "exo_app.log.1"])
    }

    @Test func aWriterThatKeepsTheFileOpenWritesFromTheStartAfterRotation() throws {
        // exo keeps writing to the handle it was started with while the app
        // empties the file; append mode stops that leaving a hole of zeros.
        let temporary = TemporaryLog(maxBytes: 4)
        let handle = try temporary.log.openForAppending()
        defer { try? handle.close() }
        try handle.write(contentsOf: Data("before rotation\n".utf8))
        try temporary.log.rotateIfNeeded()

        try handle.write(contentsOf: Data("after\n".utf8))

        #expect(temporary.contents(of: temporary.log.url) == "after\n")
    }

    @Test func textFromAnOffsetIsOnlyWhatCameAfterIt() throws {
        let temporary = TemporaryLog()
        _ = try runWithOutput(to: temporary.log, "echo previous run")
        let offset = temporary.log.size()
        _ = try runWithOutput(to: temporary.log, "echo this run")

        #expect(temporary.log.text(from: offset) == "this run\n")
        #expect(temporary.log.text(from: 0, limit: 4) == "run\n")
        // Emptied by rotation since the offset was taken: read from the start.
        #expect(temporary.log.text(from: 10_000) == "previous run\nthis run\n")
    }

    @Test func missingLogReadsAsEmpty() {
        let temporary = TemporaryLog()
        #expect(temporary.log.text(from: 0) == "")
        #expect(temporary.log.size() == 0)
        #expect(throws: Never.self) { try temporary.log.rotateIfNeeded() }
    }
}

struct ExoExitReasonTests {

    @Test func recognisesAnotherExoHoldingThePIDLock() throws {
        // What exo prints and how it exits when it can't lock ~/.exo/exo.pid.
        let temporary = TemporaryLog()
        let (reason, status) = try runWithOutput(
            to: temporary.log, "echo 'daemon already running with PID 4242' >&2; exit 1")

        let exitReason = ExoExitReason(
            terminationReason: reason, status: status, output: temporary.log.text(from: 0))

        #expect(exitReason == .alreadyRunning(pid: 4242))
        #expect(
            exitReason.message
                == "Another exo is already running on this Mac (PID 4242). Quit it, then click Restart."
        )
    }

    @Test func recognisesAnotherExoWhoseProcessIDIsUnknown() {
        let exitReason = ExoExitReason(
            terminationReason: .exit, status: 1,
            output: "daemon already running with unknown PID\n")
        #expect(exitReason == .alreadyRunning(pid: nil))
        #expect(
            exitReason.message
                == "Another exo is already running on this Mac. Quit it, then click Restart.")
    }

    @Test func otherErrorsGiveTheExitCode() {
        let exitReason = ExoExitReason(
            terminationReason: .exit, status: 1,
            output: "Traceback (most recent call last):\nValueError: boom\n")
        #expect(exitReason == .exitCode(1))
        #expect(exitReason.message == "exo quit with an error (exit code 1).")
        #expect(
            ExoExitReason(terminationReason: .exit, status: 0, output: "").message
                == "exo stopped on its own.")
    }

    @Test func aCrashIsReportedAsOne() throws {
        let temporary = TemporaryLog()
        let (reason, status) = try runWithOutput(to: temporary.log, "kill -SEGV $$")

        let exitReason = ExoExitReason(terminationReason: reason, status: status, output: "")

        #expect(exitReason == .signal(SIGSEGV))
        #expect(exitReason.message == "exo crashed (signal 11).")
        #expect(
            ExoExitReason(terminationReason: .uncaughtSignal, status: SIGKILL, output: "").message
                == "exo was killed (signal 9).")
        #expect(
            ExoExitReason(terminationReason: .uncaughtSignal, status: SIGTERM, output: "").message
                == "exo was stopped by another program (signal 15).")
    }
}
