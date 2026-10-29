import Foundation

/// exo holds an exclusive `flock` on ~/.exo/exo.pid while it runs, and a
/// second exo quits at once ("daemon already running") while it is held. So
/// the lock tells whether an exo is still running on this Mac, for example
/// one the previous run of the app stopped just before an update relaunched
/// it.
enum ExoPidLock {
    static let fileURL = ExoProcessController.exoDirectoryURL.appendingPathComponent("exo.pid")

    /// Whether another process holds the lock on `url`. Probes with a shared,
    /// non-blocking lock that is released at once.
    static func isHeld(at url: URL = fileURL) -> Bool {
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { return false }  // no PID file: no exo
        defer { close(descriptor) }
        if flock(descriptor, LOCK_SH | LOCK_NB) == 0 {
            flock(descriptor, LOCK_UN)
            return false
        }
        return errno == EWOULDBLOCK
    }

    /// Waits while `isHeld` says the lock is held, checking `every` interval,
    /// for at most `timeout`. Returns whether it was released in time.
    static func waitUntilReleased(
        timeout: Duration, every interval: Duration, isHeld: () -> Bool
    ) async -> Bool {
        let deadline = ContinuousClock.now + timeout
        while isHeld() {
            if ContinuousClock.now >= deadline || Task.isCancelled {
                return false
            }
            try? await Task.sleep(for: interval)
        }
        return true
    }
}
