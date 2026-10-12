import Foundation

/// What exo prints to stdout and stderr, saved by the app to
/// ~/.exo/exo_app.log.
///
/// exo keeps its own log in ~/.exo/exo_log/ once it is up. This file also has
/// what it prints before then, such as another exo already running, and
/// errors that never reach its logger. Each start of exo appends a header
/// line. Once the file is larger than `maxBytes` it is copied to
/// exo_app.log.1, replacing the previous copy, and emptied, so the two files
/// together stay at about twice `maxBytes`.
struct ExoLogFile: Sendable {
    static let defaultMaxBytes: UInt64 = 10 * 1024 * 1024

    let url: URL
    var maxBytes: UInt64 = Self.defaultMaxBytes

    /// The log's contents from before it was last emptied.
    var rotatedURL: URL { url.appendingPathExtension("1") }

    func size() -> UInt64 {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes?[.size] as? NSNumber)?.uint64Value ?? 0
    }

    /// Copies the log to `rotatedURL` and empties it once it is larger than
    /// `maxBytes`. It is emptied in place rather than moved because a
    /// running exo keeps writing to the file it was given. Anything exo writes
    /// between the copy and the emptying is lost. Returns whether it rotated.
    @discardableResult
    func rotateIfNeeded() throws -> Bool {
        guard size() > maxBytes else { return false }
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: rotatedURL.path) {
            try fileManager.removeItem(at: rotatedURL)
        }
        try fileManager.copyItem(at: url, to: rotatedURL)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.truncate(atOffset: 0)
        return true
    }

    /// Opens the log for exo to write to, creating it if needed. It is opened
    /// in append mode, so after `rotateIfNeeded()` has emptied it exo's writes
    /// go to the start of the file again rather than after a gap.
    func openForAppending() throws -> FileHandle {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let descriptor = open(url.path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        return FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    }

    /// What was written to the log from `offset` onwards, or only the last
    /// `limit` bytes of it. If the log has been emptied since `offset` was
    /// taken, it reads from the start.
    func text(from offset: UInt64, limit: UInt64 = 64 * 1024) -> String {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return "" }
        defer { try? handle.close() }
        guard let end = try? handle.seekToEnd() else { return "" }
        let start = max(offset <= end ? offset : 0, end > limit ? end - limit : 0)
        guard (try? handle.seek(toOffset: start)) != nil,
            let data = try? handle.readToEnd()
        else { return "" }
        return String(decoding: data, as: UTF8.self)
    }
}
