import AppKit
import Combine
import Foundation
import os.log

private let customNamespaceKey = "EXOCustomNamespace"
private let hfTokenKey = "EXOHFToken"
private let hfEndpointKey = "EXOHFEndpoint"
private let enableImageModelsKey = "EXOEnableImageModels"
private let offlineModeKey = "EXOOfflineMode"
private let fastSynchEnabledKey = "EXOFastSynchEnabled"
private let defaultModelsDirKey = "EXODefaultModelsDir"
private let additionalModelsDirsKey = "EXOAdditionalModelsDirs"
private let readOnlyModelsDirsKey = "EXOReadOnlyModelsDirs"
private let customEnvironmentVariablesKey = "EXOCustomEnvironmentVariables"

@MainActor
final class ExoProcessController: ObservableObject {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "ExoProcess")

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

    /// exo's stdout and stderr. Kept apart from exo's own log directory,
    /// ~/.exo/exo_log, which exo rotates and prunes itself.
    static let logFile = ExoLogFile(url: exoDirectoryURL.appendingPathComponent("exo_app.log"))

    /// How often the log's size is checked while exo runs.
    private static let logRotationInterval: UInt64 = 30_000_000_000

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
    @Published var defaultModelsDir: String = {
        return UserDefaults.standard.string(forKey: defaultModelsDirKey) ?? ""
    }()
    {
        didSet {
            UserDefaults.standard.set(defaultModelsDir, forKey: defaultModelsDirKey)
        }
    }
    @Published var additionalModelsDirs: String = {
        return UserDefaults.standard.string(forKey: additionalModelsDirsKey) ?? ""
    }()
    {
        didSet {
            UserDefaults.standard.set(additionalModelsDirs, forKey: additionalModelsDirsKey)
        }
    }
    @Published var readOnlyModelsDirs: String = {
        return UserDefaults.standard.string(forKey: readOnlyModelsDirsKey) ?? ""
    }()
    {
        didSet {
            UserDefaults.standard.set(readOnlyModelsDirs, forKey: readOnlyModelsDirsKey)
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

    private var process: Process?
    private var runtimeDirectoryURL: URL?
    private var pendingLaunchTask: Task<Void, Never>?
    private var logRotationTask: Task<Void, Never>?

    private var readinessTask: Task<Void, Never>?
    private let readinessProbe: () async -> Bool

    /// `readinessProbe` says whether a started exo is up yet. Until it does,
    /// the status is `.starting`.
    init(readinessProbe: @escaping () async -> Bool = { await ExoReadiness.answers() }) {
        self.readinessProbe = readinessProbe
    }

    /// Finishes once the exo that was last stopped has exited.
    private var stoppingTask: Task<Void, Never>?
    /// A start that is waiting for `stoppingTask`.
    private var launchAfterStopTask: Task<Void, Never>?

    func launchIfNeeded() {
        guard process?.isRunning != true else { return }
        launch()
    }

    func launch() {
        guard process?.isRunning != true else { return }
        // A stopped exo holds the lock on ~/.exo/exo.pid until it has exited,
        // which can take seconds, and an exo started before then quits at
        // once ("daemon already running"). So Save & Restart, or switching
        // exo off and on quickly, waits for the old one, showing Starting….
        if let stopping = stoppingTask {
            cancelPendingLaunch()
            status = .starting
            lastError = nil
            launchAfterStopTask?.cancel()
            launchAfterStopTask = Task { [weak self] in
                await stopping.value
                guard let self, !Task.isCancelled else { return }
                if self.stoppingTask == stopping {
                    self.stoppingTask = nil
                }
                self.launchAfterStopTask = nil
                self.launch()
            }
            return
        }
        do {
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

            // exo's output goes to the log so a failure can be explained. If
            // the log can't be opened, exo still runs, without it.
            let logFile = Self.logFile
            _ = try? logFile.rotateIfNeeded()
            let logHandle = try? logFile.openForAppending()
            defer { try? logHandle?.close() }
            try? logHandle?.write(
                contentsOf: Data(
                    "=== EXO.app \(buildTag()) started exo at \(Date().ISO8601Format()) ===\n".utf8)
            )
            let logOffset = logFile.size()
            child.standardOutput = logHandle ?? FileHandle.nullDevice
            child.standardError = logHandle ?? FileHandle.nullDevice

            child.terminationHandler = { [weak self] proc in
                Task { @MainActor in
                    // Ignore an exit reported after exo was started again.
                    guard let self, self.process === proc else { return }
                    self.process = nil
                    self.stopLogRotation()
                    switch self.status {
                    case .stopped:
                        break
                    case .failed:
                        break
                    default:
                        let reason = ExoExitReason(
                            terminationReason: proc.terminationReason,
                            status: proc.terminationStatus,
                            output: logFile.text(from: logOffset)
                        )
                        self.status = .failed(message: reason.message)
                        self.lastError = reason.message
                    }
                }
            }

            try child.run()
            process = child
            markRunningOnceReady(child)
            startLogRotation()
        } catch {
            process = nil
            status = .failed(message: "Couldn't start exo: \(error.localizedDescription)")
            lastError = error.localizedDescription
        }
    }

    /// Opens exo's log in Console, or shows it in Finder if Console isn't
    /// there.
    func showLogs() {
        let logURL = Self.logFile.url
        guard FileManager.default.fileExists(atPath: logURL.path) else {
            NSWorkspace.shared.activateFileViewerSelecting([Self.exoDirectoryURL])
            return
        }
        guard
            let console = NSWorkspace.shared.urlForApplication(
                withBundleIdentifier: "com.apple.Console")
        else {
            NSWorkspace.shared.activateFileViewerSelecting([logURL])
            return
        }
        NSWorkspace.shared.open(
            [logURL], withApplicationAt: console,
            configuration: NSWorkspace.OpenConfiguration()
        ) { _, error in
            if error != nil {
                DispatchQueue.main.async {
                    NSWorkspace.shared.activateFileViewerSelecting([logURL])
                }
            }
        }
    }

    /// Keeps the log to its size limit while exo runs; see `ExoLogFile`.
    private func startLogRotation() {
        stopLogRotation()
        let logFile = Self.logFile
        let interval = Self.logRotationInterval
        logRotationTask = Task.detached(priority: .utility) {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: interval)
                if Task.isCancelled { return }
                _ = try? logFile.rotateIfNeeded()
            }
        }
    }

    private func stopLogRotation() {
        logRotationTask?.cancel()
        logRotationTask = nil
    }

    /// Stays `.starting` until exo's API answers, so the app doesn't say
    /// Running while exo is still loading and the dashboard can't be opened.
    private func markRunningOnceReady(_ child: Process) {
        readinessTask?.cancel()
        let probe = readinessProbe
        readinessTask = Task { [weak self] in
            let ready = await ExoReadiness.wait(
                every: .milliseconds(500), while: { child.isRunning }, until: probe)
            guard ready, let self, self.process === child, self.status == .starting else {
                return
            }
            self.status = .running
        }
    }

    func stop() {
        readinessTask?.cancel()
        readinessTask = nil
        launchAfterStopTask?.cancel()
        launchAfterStopTask = nil
        guard let process else {
            status = .stopped
            return
        }
        process.terminationHandler = nil
        status = .stopped
        stopLogRotation()

        guard process.isRunning else {
            self.process = nil
            return
        }

        let proc = process
        self.process = nil

        stoppingTask = Task.detached {
            proc.interrupt()
            if await Self.waitForExit(of: proc, seconds: 5) { return }
            if proc.isRunning {
                proc.terminate()
            }
            if await Self.waitForExit(of: proc, seconds: 3) { return }
            if proc.isRunning {
                kill(proc.processIdentifier, SIGKILL)
            }
            _ = await Self.waitForExit(of: proc, seconds: 2)
        }
    }

    /// Waits up to `seconds` for `process` to exit, and says whether it has.
    /// Uses a deadline rather than counting sleeps, which take longer than
    /// asked on a busy Mac.
    nonisolated static func waitForExit(of process: Process, seconds: Int) async -> Bool {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while process.isRunning {
            if ContinuousClock.now >= deadline {
                return false
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
        return true
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
        environment["EXO_ZENOH_NAMESPACE"] = computeNamespace()
        if !hfToken.isEmpty {
            environment["HF_TOKEN"] = hfToken
        }
        // Also fixes up a value saved before Settings normalised it.
        let endpoint = HuggingFaceEndpoint.normalized(hfEndpoint)
        if !endpoint.isEmpty {
            environment["HF_ENDPOINT"] = endpoint
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

        let trimmedDefaultModelsDir = defaultModelsDir.trimmingCharacters(in: .whitespaces)
        if !trimmedDefaultModelsDir.isEmpty {
            environment["EXO_DEFAULT_MODELS_DIR"] = trimmedDefaultModelsDir
        }
        let trimmedAdditionalModelsDirs = additionalModelsDirs.trimmingCharacters(in: .whitespaces)
        if !trimmedAdditionalModelsDirs.isEmpty {
            environment["EXO_MODELS_DIRS"] = trimmedAdditionalModelsDirs
        }
        let trimmedReadOnlyModelsDirs = readOnlyModelsDirs.trimmingCharacters(in: .whitespaces)
        if !trimmedReadOnlyModelsDirs.isEmpty {
            environment["EXO_MODELS_READ_ONLY_DIRS"] = trimmedReadOnlyModelsDirs
        }

        // Apply user-defined arbitrary environment variables last so that
        // power users can override any of the typed fields above when
        // necessary. Empty or invalid names (e.g. stored by an older version)
        // are never passed to exo.
        for variable in customEnvironmentVariables where variable.hasInvalidName {
            Self.logger.warning(
                "Not passing custom environment variable with invalid name \"\(variable.trimmedKey, privacy: .public)\" to exo"
            )
        }
        for variable in CustomEnvironmentVariable.sanitized(customEnvironmentVariables) {
            environment[variable.key] = variable.value
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
