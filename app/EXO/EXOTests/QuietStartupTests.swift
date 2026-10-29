import Foundation
import Testing

@testable import EXO

/// A file locked the way exo locks ~/.exo/exo.pid, from another open file
/// description, as another process would.
private final class LockedFile {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("EXOTests-\(UUID().uuidString).pid")
    private var descriptor: Int32 = -1

    init() throws {
        try "123\n".write(to: url, atomically: true, encoding: .utf8)
    }

    func lock() {
        descriptor = open(url.path, O_RDWR)
        _ = flock(descriptor, LOCK_EX | LOCK_NB)
    }

    func unlock() {
        guard descriptor >= 0 else { return }
        flock(descriptor, LOCK_UN)
        close(descriptor)
        descriptor = -1
    }

    deinit {
        unlock()
        try? FileManager.default.removeItem(at: url)
    }
}

/// Says the lock is held for the first `heldFor` checks.
private final class HeldLock: @unchecked Sendable {
    private let lock = NSLock()
    private var checks = 0
    let heldFor: Int

    init(heldFor: Int) {
        self.heldFor = heldFor
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return checks
    }

    func isHeld() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        checks += 1
        return checks <= heldFor
    }
}

struct ExoPidLockTests {

    @Test func seesALockHeldByAnotherExo() throws {
        let file = try LockedFile()
        #expect(!ExoPidLock.isHeld(at: file.url))

        file.lock()
        #expect(ExoPidLock.isHeld(at: file.url))
        // Probing doesn't take the lock away.
        #expect(ExoPidLock.isHeld(at: file.url))

        file.unlock()
        #expect(!ExoPidLock.isHeld(at: file.url))
    }

    @Test func noPidFileMeansNoExo() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("EXOTests-missing-\(UUID().uuidString).pid")
        #expect(!ExoPidLock.isHeld(at: missing))
    }

    @Test func waitsUntilThePreviousExoLetsGo() async {
        let held = HeldLock(heldFor: 3)
        let released = await ExoPidLock.waitUntilReleased(
            timeout: .seconds(5), every: .milliseconds(10), isHeld: held.isHeld)
        #expect(released)
        #expect(held.count == 4)
    }

    @Test func givesUpAtTheDeadline() async {
        let held = HeldLock(heldFor: .max)
        let started = ContinuousClock.now
        let released = await ExoPidLock.waitUntilReleased(
            timeout: .milliseconds(200), every: .milliseconds(10), isHeld: held.isHeld)
        #expect(!released)
        #expect(ContinuousClock.now - started < .seconds(3))
    }
}

@MainActor
struct LaunchAtAppStartTests {

    @Test func saysStartingStraightAway() {
        let held = HeldLock(heldFor: .max)
        let controller = ExoProcessController(
            readinessProbe: { false }, previousExoIsRunning: held.isHeld)

        controller.launchWhenPreviousExoHasExited(timeout: .seconds(30))

        #expect(controller.status == .starting)
        controller.stop()
    }

    @Test func stoppingWhileWaitingForThePreviousExoStartsNothing() async throws {
        let held = HeldLock(heldFor: 5)
        let controller = ExoProcessController(
            readinessProbe: { false }, previousExoIsRunning: held.isHeld)

        controller.launchWhenPreviousExoHasExited(timeout: .seconds(30), every: .milliseconds(10))
        controller.stop()
        // Long enough for the lock to be "released" and a launch to happen.
        try await Task.sleep(for: .milliseconds(300))

        #expect(controller.status == .stopped)
    }
}

struct UpdateNotificationTests {

    @Test func oneIdentifierPerVersion() {
        #expect(UpdateNotifier.identifier(forVersion: "1.0.72") == "exo-update-1.0.72")
        #expect(
            UpdateNotifier.identifier(forVersion: "1.0.72")
                == UpdateNotifier.identifier(forVersion: "1.0.72"))
    }

    @Test func eachVersionIsAnnouncedOnce() {
        #expect(UpdateNotifier.shouldNotify(aboutVersion: "1.0.72", lastNotifiedVersion: nil))
        #expect(!UpdateNotifier.shouldNotify(aboutVersion: "1.0.72", lastNotifiedVersion: "1.0.72"))
        #expect(UpdateNotifier.shouldNotify(aboutVersion: "1.0.73", lastNotifiedVersion: "1.0.72"))
    }
}

struct NetworkSetupPromptTests {

    @Test func cancellingThePasswordPromptIsNotAFailure() {
        #expect(
            NetworkSetupError(appleScriptErrorNumber: -128, message: "User canceled.")
                == .cancelled)
        #expect(
            NetworkSetupError(appleScriptErrorNumber: 1, message: "mkdir: Permission denied")
                == .executionFailed("mkdir: Permission denied"))
        #expect(
            NetworkSetupError(appleScriptErrorNumber: nil, message: nil)
                == .executionFailed("Unknown error"))
    }

    @Test func notNowIsRemembered() {
        let suiteName = FileManager.default.temporaryDirectory
            .appendingPathComponent("EXOTests-\(UUID().uuidString)").path
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        #expect(!NetworkSetupHelper.setupWasDeclined(defaults: defaults))
        NetworkSetupHelper.recordSetupDeclined(true, defaults: defaults)
        #expect(NetworkSetupHelper.setupWasDeclined(defaults: defaults))
        NetworkSetupHelper.recordSetupDeclined(false, defaults: defaults)
        #expect(!NetworkSetupHelper.setupWasDeclined(defaults: defaults))
    }
}
