import AppKit
import Combine
import Foundation

private let customNamespaceKey = "EXOCustomNamespace"
private let hfTokenKey = "EXOHFToken"
private let hfEndpointKey = "EXOHFEndpoint"
private let enableImageModelsKey = "EXOEnableImageModels"
private let offlineModeKey = "EXOOfflineMode"
private let fastSynchEnabledKey = "EXOFastSynchEnabled"
private let onboardingCompletedKey = "EXOOnboardingCompleted"
private let customEnvironmentVariablesKey = "EXOCustomEnvironmentVariables"

/// A user-defined environment variable that is injected into the exo child
/// process at launch. Used to pass arbitrary key/value settings to exo
/// without having to add first-class UI for each one.
struct CustomEnvironmentVariable: Codable, Identifiable, Equatable {
    var id: UUID
    var key: String
    var value: String

    init(id: UUID = UUID(), key: String = "", value: String = "") {
        self.id = id
        self.key = key
        self.value = value
    }
}

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
    @Published var hfToken: String = {
        return UserDefaults.standard.string(forKey: hfTokenKey) ?? ""
    }()
    {
        didSet {
            UserDefaults.standard.set(hfToken, forKey: hfTokenKey)
        }
    }
    @Published var hfEndpoint: String = {
        return UserDefaults.standard.string(forKey: hfEndpointKey) ?? ""
    }()
    {
        didSet {
            UserDefaults.standard.set(hfEndpoint, forKey: hfEndpointKey)
        }
    }
    @Published var enableImageModels: Bool = {
        return UserDefaults.standard.bool(forKey: enableImageModelsKey)
    }()
    {
        didSet {
            UserDefaults.standard.set(enableImageModels, forKey: enableImageModelsKey)
        }
    }
    @Published var offlineMode: Bool = {
        return UserDefaults.standard.bool(forKey: offlineModeKey)
    }()
    {
        didSet {
            UserDefaults.standard.set(offlineMode, forKey: offlineModeKey)
        }
    }
    @Published var fastSynchEnabled: Bool = {
        if UserDefaults.standard.object(forKey: fastSynchEnabledKey) == nil {
            return true
        }
        return UserDefaults.standard.bool(forKey: fastSynchEnabledKey)
    }()
    {
        didSet {
            UserDefaults.standard.set(fastSynchEnabled, forKey: fastSynchEnabledKey)
        }
    }
    @Published var customEnvironmentVariables: [CustomEnvironmentVariable] = {
        guard
            let data = UserDefaults.standard.data(forKey: customEnvironmentVariablesKey),
            let decoded = try? JSONDecoder().decode(
                [CustomEnvironmentVariable].self, from: data
            )
        else {
            return []
        }
        return decoded
    }()
    {
        didSet {
            guard let data = try? JSONEncoder().encode(customEnvironmentVariables) else {
                return
            }
            UserDefaults.standard.set(data, forKey: customEnvironmentVariablesKey)
        }
    }

    /// Fires once when EXO transitions to `.running` for the very first time (fresh install).
    @Published private(set) var isFirstLaunchReady = false

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

            // Show welcome popout on every launch
            isFirstLaunchReady = true
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
        status = .stopped

        guard process.isRunning else {
            self.process = nil
            return
        }

        let proc = process
        self.process = nil

        Task.detached {
            proc.interrupt()

            for _ in 0..<50 {
                if !proc.isRunning { return }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }

            if proc.isRunning {
                proc.terminate()
            }

            for _ in 0..<30 {
                if !proc.isRunning { return }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }

            if proc.isRunning {
                kill(proc.processIdentifier, SIGKILL)
            }
        }
    }

    func restart() {
        stop()
        launch()
    }

    /// Mark onboarding as completed (user interacted with the welcome popout).
    func markOnboardingCompleted() {
        UserDefaults.standard.set(true, forKey: onboardingCompletedKey)
    }

    /// Reset onboarding so the welcome popout appears on next launch.
    func resetOnboarding() {
        UserDefaults.standard.removeObject(forKey: onboardingCompletedKey)
        isFirstLaunchReady = false
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
        if !hfEndpoint.isEmpty {
            environment["HF_ENDPOINT"] = hfEndpoint
        }
        if enableImageModels {
            environment["EXO_ENABLE_IMAGE_MODELS"] = "true"
        }
        if offlineMode {
            environment["EXO_OFFLINE"] = "true"
        }
        environment["EXO_FAST_SYNCH"] = fastSynchEnabled ? "true" : "false"

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

        // Apply user-defined arbitrary environment variables last so that
        // power users can override any of the built-in keys above when
        // necessary. Empty keys are ignored.
        for variable in customEnvironmentVariables {
            let trimmedKey = variable.key.trimmingCharacters(in: .whitespaces)
            guard !trimmedKey.isEmpty else { continue }
            environment[trimmedKey] = variable.value
        }

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
