import AppKit
import Combine
import Foundation

private let customNamespaceKey = "EXOCustomNamespace"

@MainActor
final class ExoProcessController: ObservableObject {
    enum Status: Equatable {
        case stopped
        case starting
        case running
        case failed(message: String)

        var displayText: String {
            switch self {
            case .stopped:
                return "Stopped"
            case .starting:
                return "Starting…"
            case .running:
                return "Running"
            case .failed:
                return "Failed"
            }
        }
    }

    @Published private(set) var status: Status = .stopped
    @Published private(set) var lastError: String?
    @Published private(set) var launchCountdownSeconds: Int?
    @Published var customNamespace: String = {
        return UserDefaults.standard.string(forKey: customNamespaceKey) ?? ""
    }()
    {
        didSet {
            UserDefaults.standard.set(customNamespace, forKey: customNamespaceKey)
        }
    }

    private var process: Process?
    private var runtimeDirectoryURL: URL?
    private var pendingLaunchTask: Task<Void, Never>?

    func launchIfNeeded() {
        guard process?.isRunning != true else { return }
        launch()
    }

    func launch() {
        do {
            guard process?.isRunning != true else { return }
            cancelPendingLaunch()
            status = .starting
            lastError = nil
            let runtimeURL = try resolveRuntimeDirectory()
            runtimeDirectoryURL = runtimeURL

            let executableURL = runtimeURL.appendingPathComponent("exo")

            let child = Process()
            child.executableURL = executableURL
            child.currentDirectoryURL = runtimeURL
            child.environment = makeEnvironment(for: runtimeURL)

            child.standardOutput = FileHandle.nullDevice
            child.standardError = FileHandle.nullDevice

            child.terminationHandler = { [weak self] proc in
                Task { @MainActor in
                    guard let self else { return }
                    self.process = nil
                    switch self.status {
                    case .stopped:
                        break
                    case .failed:
                        break
                    default:
                        self.status = .failed(
                            message: "Exited with code \(proc.terminationStatus)"
                        )
                        self.lastError = "Process exited with code \(proc.terminationStatus)"
                    }
                }
            }

            try child.run()
            process = child
            status = .running
        } catch {
            process = nil
            status = .failed(message: "Launch error")
            lastError = error.localizedDescription
        }
    }

    func stop() {
        guard let process else {
            status = .stopped
            return
        }
        process.terminationHandler = nil
        if process.isRunning {
            process.terminate()
        }
        self.process = nil
        status = .stopped
    }

    func restart() {
        stop()
        launch()
    }

    func scheduleLaunch(after seconds: TimeInterval) {
        cancelPendingLaunch()
        let start = max(1, Int(ceil(seconds)))
        pendingLaunchTask = Task { [weak self] in
            guard let self else { return }
            await MainActor.run {
                self.launchCountdownSeconds = start
            }
            var remaining = start
            while remaining > 0 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                remaining -= 1
                if Task.isCancelled { return }
                await MainActor.run {
                    if remaining > 0 {
                        self.launchCountdownSeconds = remaining
                    } else {
                        self.launchCountdownSeconds = nil
                        self.launchIfNeeded()
                    }
                }
            }
        }
    }

    func cancelPendingLaunch() {
        pendingLaunchTask?.cancel()
        pendingLaunchTask = nil
        launchCountdownSeconds = nil
    }

    func revealRuntimeDirectory() {
        guard let runtimeDirectoryURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([runtimeDirectoryURL])
    }

    func statusTintColor() -> NSColor {
        switch status {
        case .running:
            return .systemGreen
        case .starting:
            return .systemYellow
        case .failed:
            return .systemRed
        case .stopped:
            return .systemGray
        }
    }

    private func resolveRuntimeDirectory() throws -> URL {
        let fileManager = FileManager.default

        if let override = ProcessInfo.processInfo.environment["EXO_RUNTIME_DIR"] {
            let url = URL(fileURLWithPath: override).standardizedFileURL
            if fileManager.fileExists(atPath: url.path) {
                return url
            }
        }

        if let resourceRoot = Bundle.main.resourceURL {
            let bundled = resourceRoot.appendingPathComponent("exo", isDirectory: true)
            if fileManager.fileExists(atPath: bundled.path) {
                return bundled
            }
        }

        let repoCandidate = URL(fileURLWithPath: fileManager.currentDirectoryPath)
            .appendingPathComponent("dist/exo", isDirectory: true)
        if fileManager.fileExists(atPath: repoCandidate.path) {
            return repoCandidate
        }

        throw RuntimeError("Unable to locate the packaged EXO runtime.")
    }

    private func makeEnvironment(for runtimeURL: URL) -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        environment["EXO_RUNTIME_DIR"] = runtimeURL.path
        environment["EXO_LIBP2P_NAMESPACE"] = computeNamespace()

        var paths: [String] = []
        if let existing = environment["PATH"], !existing.isEmpty {
            paths = existing.split(separator: ":").map(String.init)
        }

        let required = [
            runtimeURL.path,
            runtimeURL.appendingPathComponent("_internal").path,
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ]

        for entry in required.reversed() {
            if !paths.contains(entry) {
                paths.insert(entry, at: 0)
            }
        }

        environment["PATH"] = paths.joined(separator: ":")
        return environment
    }

    private func buildTag() -> String {
        if let tag = Bundle.main.infoDictionary?["EXOBuildTag"] as? String, !tag.isEmpty {
            return tag
        }
        if let short = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
            !short.isEmpty
        {
            return short
        }
        return "dev"
    }

    private func computeNamespace() -> String {
        let base = buildTag()
        let custom = customNamespace.trimmingCharacters(in: .whitespaces)
        return custom.isEmpty ? base : custom
    }
}

struct RuntimeError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? {
        message
    }
}
