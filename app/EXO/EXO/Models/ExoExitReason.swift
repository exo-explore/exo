import Foundation

/// Why exo exited when the app hadn't asked it to, worked out from how the
/// process ended and what it printed, so the popover can say it in words.
enum ExoExitReason: Equatable {
    /// exo found another exo already running on this Mac (one holds a lock
    /// on ~/.exo/exo.pid) and quit.
    case alreadyRunning(pid: Int32?)
    /// exo was ended by a signal: it crashed, or something killed it.
    case signal(Int32)
    case exitCode(Int32)

    /// `output` is what exo printed before it exited. When it can't lock its
    /// PID file it prints "daemon already running with PID 123" (or "with
    /// unknown PID") and exits with code 1.
    init(terminationReason: Process.TerminationReason, status: Int32, output: String) {
        if terminationReason == .uncaughtSignal {
            self = .signal(status)
        } else if status != 0,
            let marker = output.range(of: "daemon already running with ", options: .backwards)
        {
            let rest = output[marker.upperBound...]
            let pid =
                rest.hasPrefix("PID ") ? Int32(rest.dropFirst(4).prefix(while: \.isNumber)) : nil
            self = .alreadyRunning(pid: pid)
        } else {
            self = .exitCode(status)
        }
    }

    var message: String {
        switch self {
        case .alreadyRunning(let pid):
            let which = pid.map { " (PID \($0))" } ?? ""
            return
                "Another exo is already running on this Mac\(which). Quit it, then click Restart."
        case .signal(let signal) where signal == SIGKILL:
            return "exo was killed (signal \(signal))."
        case .signal(let signal) where [SIGTERM, SIGINT, SIGHUP].contains(signal):
            return "exo was stopped by another program (signal \(signal))."
        case .signal(let signal):
            return "exo crashed (signal \(signal))."
        case .exitCode(0):
            return "exo stopped on its own."
        case .exitCode(let code):
            return "exo quit with an error (exit code \(code))."
        }
    }
}
