import AppKit
import Combine
import Foundation
import os.log

private let customNamespaceKey = "EXOCustomNamespace"
private let hfEndpointKey = "EXOHFEndpoint"
private let enableImageModelsKey = "EXOEnableImageModels"
private let offlineModeKey = "EXOOfflineMode"
private let defaultModelsDirKey = "EXODefaultModelsDir"
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

    nonisolated static let exoDirectoryURL: URL = {
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".exo")
    }()

    /// exo's stdout and stderr. Kept apart from exo's own log directory,
    /// ~/.exo/exo_log, which exo rotates and prunes itself.
    nonisolated static let logFile = ExoLogFile(
        url: exoDirectoryURL.appendingPathComponent("exo_app.log"))

    /// How often the log's size is checked while exo runs.
    private static let logRotationInterval: UInt64 = 30_000_000_000

    @Published private(set) var status: Status = .stopped
    @Published private(set) var lastError: String?
    /// Whether the exo that failed had been running (answering) first: a
    /// crash, rather than a failed start. Set before `status` changes.
    private(set) var failedAfterRunning = false
    /// Whether the current exo has answered since it was started.
    private var hasAnsweredSinceLaunch = false
    // MARK: Settings that exo reads when it starts

    private let defaults: UserDefaults
    /// Where the Hugging Face token is kept; see `HuggingFaceToken`.
    private let tokenKeychain: KeychainPassword

    @Published var customNamespace: String {
        didSet {
            defaults.set(customNamespace, forKey: customNamespaceKey)
            settingsDidChange()
        }
    }
    @Published var hfToken: String {
        didSet {
            do {
                try HuggingFaceToken.save(hfToken, to: tokenKeychain)
            } catch {
                Self.logger.error(
                    "Couldn't save the Hugging Face token: \(error.localizedDescription, privacy: .public)"
                )
            }
            settingsDidChange()
        }
    }
    @Published var hfEndpoint: String {
        didSet {
            defaults.set(hfEndpoint, forKey: hfEndpointKey)
            settingsDidChange()
        }
    }
    @Published var enableImageModels: Bool {
        didSet {
            defaults.set(enableImageModels, forKey: enableImageModelsKey)
            settingsDidChange()
        }
    }
    @Published var offlineMode: Bool {
        didSet {
            defaults.set(offlineMode, forKey: offlineModeKey)
            settingsDidChange()
        }
    }
    @Published var fastSynch: FastSynchSetting {
        didSet {
            defaults.set(fastSynch.rawValue, forKey: FastSynchSetting.storageKey)
            settingsDidChange()
        }
    }
    /// Where exo downloads models. Empty means its default, ~/.exo/models.
    @Published var defaultModelsDir: String {
        didSet {
            defaults.set(defaultModelsDir, forKey: defaultModelsDirKey)
            settingsDidChange()
        }
    }
    @Published var otherModelFolders: [ModelFolder] {
        didSet {
            ModelFolder.save(otherModelFolders, to: defaults)
            settingsDidChange()
        }
    }
    @Published var customEnvironmentVariables: [CustomEnvironmentVariable] {
        didSet {
            if let data = try? JSONEncoder().encode(customEnvironmentVariables) {
                defaults.set(data, forKey: customEnvironmentVariablesKey)
            }
            settingsDidChange()
        }
    }

    /// The settings as exo would get them if it started now.
    var launchSettings: ExoLaunchSettings {
        ExoLaunchSettings(
            clusterName: customNamespace,
            huggingFaceToken: hfToken,
            huggingFaceEndpoint: hfEndpoint,
            showImageModels: enableImageModels,
            offline: offlineMode,
            fastSynch: fastSynch,
            downloadFolder: defaultModelsDir,
            otherModelFolders: otherModelFolders,
            customEnvironmentVariables: customEnvironmentVariables
        )
    }

    /// Whether exo runs with older settings than these and restarting it
    /// would stop models running on this Mac, so the user is asked first.
    /// When nothing on this Mac would stop, a change restarts exo by itself.
    @Published private(set) var needsRestart = false

    /// Whether a model has a part running on this Mac, or nil if that isn't
    /// known. The app sets it from the cluster state.
    var modelsRunningOnThisMac: () -> Bool? = { nil }

    /// How long settings must stay unchanged before they are applied, so
    /// that several quick changes restart exo once.
    static let settingsApplyDelay: Duration = .seconds(1)

    /// The settings environment the running exo was started with.
    private var runningSettingsEnvironment: [String: String]?
    private var applySettingsTask: Task<Void, Never>?
    /// Settings text fields being edited. Changes are applied once the user
    /// has left them, so exo doesn't restart between two fields.
    private var settingsFieldsBeingEdited = Set<UUID>()

    private var process: Process?
    private var runtimeDirectoryURL: URL?
    private var pendingLaunchTask: Task<Void, Never>?
    private var logRotationTask: Task<Void, Never>?

    private var readinessTask: Task<Void, Never>?
    private let readinessProbe: () async -> Bool
    private let previousExoIsRunning: () -> Bool

    /// `readinessProbe` says whether a started exo is up yet. Until it does,
    /// the status is `.starting`. `previousExoIsRunning` says whether an exo
    /// this app didn't start still holds exo's PID lock.
    init(
        readinessProbe: @escaping () async -> Bool = { await ExoReadiness.answers() },
        previousExoIsRunning: @escaping () -> Bool = { ExoPidLock.isHeld() },
        defaults: UserDefaults = .standard,
        tokenKeychain: KeychainPassword = HuggingFaceToken.keychain
    ) {
        self.readinessProbe = readinessProbe
        self.previousExoIsRunning = previousExoIsRunning
        self.defaults = defaults
        self.tokenKeychain = tokenKeychain
        customNamespace = defaults.string(forKey: customNamespaceKey) ?? ""
        hfToken = HuggingFaceToken.load(from: tokenKeychain, migrating: defaults)
        hfEndpoint = defaults.string(forKey: hfEndpointKey) ?? ""
        enableImageModels = defaults.bool(forKey: enableImageModelsKey)
        offlineMode = defaults.bool(forKey: offlineModeKey)
        fastSynch = FastSynchSetting.load(from: defaults)
        defaultModelsDir = defaults.string(forKey: defaultModelsDirKey) ?? ""
        otherModelFolders = ModelFolder.load(from: defaults)
        customEnvironmentVariables =
            defaults.data(forKey: customEnvironmentVariablesKey).flatMap {
                try? JSONDecoder().decode([CustomEnvironmentVariable].self, from: $0)
            } ?? []
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
            hasAnsweredSinceLaunch = false
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
            let settingsEnvironment = launchSettings.environment(buildTag: buildTag())
            for variable in customEnvironmentVariables where variable.hasInvalidName {
                Self.logger.warning(
                    "Not passing custom environment variable with invalid name \"\(variable.trimmedKey, privacy: .public)\" to exo"
                )
            }
            child.environment = Self.makeEnvironment(
                base: ProcessInfo.processInfo.environment,
                runtimeURL: runtimeURL,
                settingsEnvironment: settingsEnvironment)

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
                        self.failedAfterRunning = self.hasAnsweredSinceLaunch
                        self.status = .failed(message: reason.message)
                        self.lastError = reason.message
                    }
                    self.runningWithSettings(nil)
                }
            }

            try child.run()
            process = child
            runningWithSettings(settingsEnvironment)
            markRunningOnceReady(child)
            startLogRotation()
        } catch {
            process = nil
            failedAfterRunning = false
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
            self.hasAnsweredSinceLaunch = true
            self.status = .running
        }
    }

    func stop() {
        cancelPendingLaunch()
        runningWithSettings(nil)
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

    /// Stops exo, and returns once it has exited (or been killed).
    func stopAndWaitForExit() async {
        stop()
        await stoppingTask?.value
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

    // MARK: Applying settings

    /// Restarts exo so it uses the current settings. Running models stop.
    func restartToApplySettings() {
        restart()
    }

    /// Tells the controller that a settings text field started or stopped
    /// being edited.
    func settingsField(_ field: UUID, isEditing: Bool) {
        if isEditing {
            settingsFieldsBeingEdited.insert(field)
        } else if settingsFieldsBeingEdited.remove(field) != nil {
            settingsDidChange()
        }
    }

    private func settingsDidChange() {
        applySettingsTask?.cancel()
        applySettingsTask = Task { [weak self] in
            try? await Task.sleep(for: Self.settingsApplyDelay)
            guard !Task.isCancelled else { return }
            self?.applySettings()
        }
    }

    /// Restarts exo with the changed settings if nothing on this Mac would
    /// stop; otherwise asks first, through `needsRestart`.
    private func applySettings() {
        guard settingsFieldsBeingEdited.isEmpty else { return }
        let decision = SettingsRestartPolicy.decide(
            status: status,
            runningEnvironment: runningSettingsEnvironment,
            settingsEnvironment: launchSettings.environment(buildTag: buildTag()),
            modelsRunningOnThisMac: modelsRunningOnThisMac()
        )
        switch decision {
        case .nothing:
            needsRestart = false
        case .restartNow:
            Self.logger.info("Restarting exo to apply changed settings")
            restart()
        case .ask:
            needsRestart = true
        }
    }

    /// Records the settings exo was started with, or nil once it has stopped.
    private func runningWithSettings(_ environment: [String: String]?) {
        runningSettingsEnvironment = environment
        needsRestart = false
    }

    /// Starts exo as the app starts, saying "Starting…" straight away. An exo
    /// from the previous run of the app, stopped just before the app quit
    /// (e.g. to install an update), can still hold exo's PID lock while it
    /// shuts down, and a new exo quits at once while it does. So this waits
    /// for that exo first, for up to `timeout`; usually there is none and exo
    /// starts immediately.
    func launchWhenPreviousExoHasExited(
        timeout: Duration = .seconds(10), every interval: Duration = .milliseconds(250)
    ) {
        guard process?.isRunning != true else { return }
        cancelPendingLaunch()
        status = .starting
        lastError = nil
        let previousExoIsRunning = previousExoIsRunning
        pendingLaunchTask = Task { [weak self] in
            _ = await ExoPidLock.waitUntilReleased(
                timeout: timeout, every: interval, isHeld: previousExoIsRunning)
            guard let self, !Task.isCancelled else { return }
            self.pendingLaunchTask = nil
            self.launch()
        }
    }

    func cancelPendingLaunch() {
        pendingLaunchTask?.cancel()
        pendingLaunchTask = nil
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

    /// exo's environment: the app's own, with the runtime's folders first on
    /// PATH, and `settingsEnvironment` (see `ExoLaunchSettings`) over it.
    nonisolated static func makeEnvironment(
        base: [String: String],
        runtimeURL: URL,
        settingsEnvironment: [String: String]
    ) -> [String: String] {
        var environment = base
        environment["EXO_RUNTIME_DIR"] = runtimeURL.path

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

        // Last, so that custom variables can override anything above.
        environment.merge(settingsEnvironment) { _, setting in setting }
        return environment
    }

    private func buildTag() -> String {
        Self.buildTag
    }

    /// This build's version, which is also exo's cluster name when none is
    /// set.
    nonisolated static var buildTag: String {
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
