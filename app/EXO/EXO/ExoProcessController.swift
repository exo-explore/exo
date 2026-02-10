import AppKit
import Combine
import Foundation

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

    static let exoDirectoryURL: URL = {
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".exo")
    }()

    private static let settingsFileURL: URL = {
        exoDirectoryURL.appendingPathComponent("settings.json")
    }()

    @Published private(set) var status: Status = .stopped
    @Published private(set) var lastError: String?
    @Published private(set) var launchCountdownSeconds: Int?
    @Published var customNamespace: String = "" {
        didSet { saveCurrentSettings() }
    }
    @Published var hfToken: String = "" {
        didSet { saveCurrentSettings() }
    }
    @Published var enableImageModels: Bool = false {
        didSet { saveCurrentSettings() }
    }

    private var process: Process?
    private var runtimeDirectoryURL: URL?
    private var pendingLaunchTask: Task<Void, Never>?

    init() {
        let settings = Self.loadSettings()
        self.customNamespace = settings.customNamespace
        self.hfToken = settings.hfToken
        self.enableImageModels = settings.enableImageModels
    }

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
            let exoHomeURL = Self.exoDirectoryURL
            try? FileManager.default.createDirectory(
                at: exoHomeURL, withIntermediateDirectories: true
            )
            child.currentDirectoryURL = exoHomeURL
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
        if !hfToken.isEmpty {
            environment["HF_TOKEN"] = hfToken
        }
        if enableImageModels {
            environment["EXO_ENABLE_IMAGE_MODELS"] = "true"
        }

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

    // MARK: - Settings persistence (~/.exo/settings.json)

    private struct ExoSettings: Codable {
        var customNamespace: String = ""
        var hfToken: String = ""
        var enableImageModels: Bool = false
    }

    private static func loadSettings() -> ExoSettings {
        if let data = try? Data(contentsOf: settingsFileURL),
            let settings = try? JSONDecoder().decode(ExoSettings.self, from: data)
        {
            return settings
        }
        return ExoSettings()
    }

    private static func saveSettings(_ settings: ExoSettings) {
        try? FileManager.default.createDirectory(
            at: exoDirectoryURL, withIntermediateDirectories: true
        )
        guard let data = try? JSONEncoder().encode(settings) else { return }
        try? data.write(to: settingsFileURL, options: .atomic)
    }

    private func saveCurrentSettings() {
        let settings = ExoSettings(
            customNamespace: customNamespace,
            hfToken: hfToken,
            enableImageModels: enableImageModels
        )
        Self.saveSettings(settings)
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
