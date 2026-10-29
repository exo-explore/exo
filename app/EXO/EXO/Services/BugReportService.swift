import Foundation

struct BugReportOutcome: Equatable {
    let success: Bool
    let message: String
}

enum BugReportError: LocalizedError {
    case invalidEndpoint
    case presignedUrlFailed(String)
    case uploadFailed(String)
    case collectFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint:
            return "Bug report endpoint is invalid."
        case .presignedUrlFailed(let message):
            return "Failed to get presigned URLs: \(message)"
        case .uploadFailed(let message):
            return "Bug report upload failed: \(message)"
        case .collectFailed(let message):
            return "Bug report collection failed: \(message)"
        }
    }
}

/// One file of a bug report, at its path inside the report.
struct BugReportFile: Equatable {
    let path: String
    let data: Data
}

/// Collects a bug report and sends it, or writes it to a folder so the user
/// can see exactly what would be sent.
///
/// A report holds only what is needed to diagnose a problem, with the user's
/// prompts, the names of the user and their devices, and hardware addresses
/// taken out (see `BugReportRedactor`):
/// - EXO's logs: exo's own (~/.exo/exo_log) and its output kept by the app
///   (~/.exo/exo_app.log), which says why exo couldn't start or stopped;
/// - the cluster's state and recent events, as exo reports them;
/// - this Mac's model and macOS version and its network setup
///   (`BugReportSystemInfo`), plus `ifconfig`, Thunderbolt Bridge and RDMA
///   status;
/// - the user's description.
struct BugReportService {
    private struct PresignedUrlsRequest: Codable {
        let keys: [String]
    }

    private struct PresignedUrlsResponse: Codable {
        let urls: [String: String]
        let expiresIn: Int?
    }

    var redactor = BugReportRedactor.current

    func sendReport(
        baseURL: URL = URL(string: "http://127.0.0.1:52415")!,
        now: Date = Date(),
        isManual: Bool = false,
        userDescription: String? = nil
    ) async throws -> BugReportOutcome {
        let timestamp = Self.runTimestampString(now)
        let dayPrefix = Self.dayPrefixString(now)
        let prefix = "reports/\(dayPrefix)/\(timestamp)/"

        let files = try await collect(
            baseURL: baseURL, now: now, isManual: isManual, userDescription: userDescription)
        guard !files.isEmpty else {
            return BugReportOutcome(success: false, message: "No data to upload")
        }

        let uploadItems = files.map { (key: "\(prefix)\($0.path)", body: $0.data) }
        let presignedUrls = try await fetchPresignedUploadUrls(keys: uploadItems.map(\.key))
        for item in uploadItems {
            guard let urlString = presignedUrls[item.key], let url = URL(string: urlString) else {
                throw BugReportError.uploadFailed("Missing presigned URL for \(item.key)")
            }
            try await uploadToPresignedUrl(url: url, body: item.body)
        }

        return BugReportOutcome(success: true, message: "Report sent. Thank you.")
    }

    /// Everything a report sends.
    func collect(
        baseURL: URL = URL(string: "http://127.0.0.1:52415")!,
        now: Date = Date(),
        isManual: Bool = false,
        userDescription: String? = nil
    ) async throws -> [BugReportFile] {
        let ifconfigText = redactor.scrub(try captureIfconfig())
        let debugInfo = readDebugInfo()

        let stateData = await fetch(url: baseURL.appendingPathComponent("state"))
        // The API keeps its recent events in memory; nothing writes them to disk
        let eventsData = await fetch(url: baseURL.appendingPathComponent("events"))

        let reportJSON = makeReportJson(
            timestamp: Self.runTimestampString(now),
            ifconfig: ifconfigText,
            debugInfo: debugInfo,
            isManual: isManual,
            clusterTbBridgeStatus: extractClusterTbBridgeStatus(from: stateData),
            userDescription: userDescription
        )

        var files = readLogs()
        // Sent only if they could be checked (see `BugReportRedactor.scrubJSON`).
        if let state = stateData.flatMap(redactor.scrubJSON) {
            files.append(BugReportFile(path: "state.json", data: state))
        }
        if let events = eventsData.flatMap(redactor.scrubJSON) {
            files.append(BugReportFile(path: "events.json", data: events))
        }
        if let reportJSON {
            files.append(BugReportFile(path: "report.json", data: reportJSON))
        }
        return files
    }

    /// Writes `files` into `directory`, creating it and the folders inside.
    static func write(_ files: [BugReportFile], to directory: URL) throws {
        let fileManager = FileManager.default
        for file in files {
            let url = directory.appendingPathComponent(file.path)
            try fileManager.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try file.data.write(to: url)
        }
    }

    private static func dayPrefixString(_ date: Date) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .current
        let components = calendar.dateComponents([.year, .month, .day], from: date)
        let year = components.year ?? 0
        let month = components.month ?? 0
        let day = components.day ?? 0
        return String(format: "%04d/%02d/%02d", year, month, day)
    }

    static func runTimestampString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0) ?? .current
        formatter.dateFormat = "yyyy-MM-dd'T'HHmmss.SSS'Z'"
        return formatter.string(from: date)
    }

    private func fetchPresignedUploadUrls(keys: [String], bundle: Bundle = .main) async throws
        -> [String: String]
    {
        guard
            let endpointString = bundle.infoDictionary?["EXOBugReportPresignedUrlEndpoint"]
                as? String
        else {
            throw BugReportError.invalidEndpoint
        }
        let trimmedEndpointString = endpointString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedEndpointString.isEmpty, let endpoint = URL(string: trimmedEndpointString)
        else {
            throw BugReportError.invalidEndpoint
        }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 10
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let encoder = JSONEncoder()
        request.httpBody = try encoder.encode(PresignedUrlsRequest(keys: keys))

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw BugReportError.presignedUrlFailed("Non-HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw BugReportError.presignedUrlFailed("HTTP status \(http.statusCode)")
        }

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PresignedUrlsResponse.self, from: data)
        return decoded.urls
    }

    /// exo's own log of this run, and its output kept by the app (with the
    /// copy kept when that was last emptied), which has earlier runs too and
    /// says why exo couldn't start. exo's compressed logs of earlier runs are
    /// left out: they can't be checked for prompts here, and the app's copy
    /// covers those runs.
    private func readLogs() -> [BugReportFile] {
        let exoDirectory = ExoProcessController.exoDirectoryURL
        let logDirectory = exoDirectory.appendingPathComponent("exo_log")
        var urls: [(path: String, url: URL)] = []
        let names = (try? FileManager.default.contentsOfDirectory(atPath: logDirectory.path)) ?? []
        for name in names.sorted() where !name.hasSuffix(".zst") {
            urls.append(("exo_log/\(name)", logDirectory.appendingPathComponent(name)))
        }
        let appLog = ExoProcessController.logFile
        urls.append((appLog.url.lastPathComponent, appLog.url))
        urls.append((appLog.rotatedURL.lastPathComponent, appLog.rotatedURL))

        return urls.compactMap { entry in
            guard let data = try? Data(contentsOf: entry.url) else { return nil }
            let text = redactor.scrub(String(decoding: data, as: UTF8.self))
            return BugReportFile(path: entry.path, data: Data(text.utf8))
        }
    }

    private func captureIfconfig() throws -> String {
        let result = runCommand(["/sbin/ifconfig"])
        guard result.exitCode == 0 else {
            throw BugReportError.collectFailed(
                result.error.isEmpty ? "ifconfig failed" : result.error)
        }
        return result.output
    }

    private func readDebugInfo() -> DebugInfo {
        DebugInfo(
            thunderboltBridgeDisabled: readThunderboltBridgeDisabled(),
            interfaces: readInterfaces(),
            rdma: readRDMADebugInfo()
        )
    }

    private func readRDMADebugInfo() -> DebugInfo.RDMADebugInfo {
        DebugInfo.RDMADebugInfo(
            rdmaCtlStatus: safeRunCommand(["/usr/bin/rdma_ctl", "status"]),
            ibvDevices: safeRunCommand(["/usr/bin/ibv_devices"]).map(redactor.scrub),
            ibvDevinfo: safeRunCommand(["/usr/bin/ibv_devinfo"]).map(redactor.scrub)
        )
    }

    private func readThunderboltBridgeDisabled() -> Bool? {
        // Dynamically find the Thunderbolt Bridge service (don't assume the name)
        guard let serviceName = ThunderboltBridgeDetector.findThunderboltBridgeServiceName() else {
            // No bridge containing Thunderbolt interfaces exists
            return nil
        }

        guard let isEnabled = ThunderboltBridgeDetector.isServiceEnabled(serviceName: serviceName)
        else {
            return nil
        }

        // Return true if disabled, false if enabled
        return !isEnabled
    }

    private func readInterfaces() -> [DebugInfo.InterfaceStatus] {
        (0...7).map { "en\($0)" }.map { iface in
            let result = runCommand(["/sbin/ifconfig", iface])
            guard result.exitCode == 0 else {
                return DebugInfo.InterfaceStatus(name: iface, ip: nil)
            }
            let ip = firstInet(from: result.output)
            return DebugInfo.InterfaceStatus(name: iface, ip: ip)
        }
    }

    private func firstInet(from ifconfigOutput: String) -> String? {
        for line in ifconfigOutput.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("inet ") else { continue }
            let parts = trimmed.split(separator: " ")
            if parts.count >= 2 {
                let candidate = String(parts[1])
                if candidate != "127.0.0.1" {
                    return candidate
                }
            }
        }
        return nil
    }

    private func fetch(url: URL) async -> Data? {
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode)
            else {
                return nil
            }
            return data
        } catch {
            return nil
        }
    }

    private func uploadToPresignedUrl(url: URL, body: Data) async throws {
        let maxAttempts = 2
        var lastError: Error?

        for attempt in 1...maxAttempts {
            do {
                var request = URLRequest(url: url)
                request.httpMethod = "PUT"
                request.httpBody = body
                request.timeoutInterval = 30

                let (_, response) = try await URLSession.shared.data(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw BugReportError.uploadFailed("Non-HTTP response")
                }
                guard (200..<300).contains(http.statusCode) else {
                    throw BugReportError.uploadFailed("HTTP status \(http.statusCode)")
                }
                return
            } catch {
                lastError = error
                if attempt < maxAttempts {
                    try await Task.sleep(nanoseconds: 400_000_000)
                }
            }
        }

        throw BugReportError.uploadFailed(lastError?.localizedDescription ?? "Unknown error")
    }

    private func makeReportJson(
        timestamp: String,
        ifconfig: String,
        debugInfo: DebugInfo,
        isManual: Bool,
        clusterTbBridgeStatus: [[String: Any]]?,
        userDescription: String? = nil
    ) -> Data? {
        let exo = readExoMetadata()
        var payload: [String: Any] = [
            "timestamp": timestamp,
            "ifconfig": ifconfig,
            "debug": debugInfo.toDictionary(),
            "system": BugReportSystemInfo.collect(run: safeRunCommand),
            "exo_version": exo.version as Any,
            "exo_commit": exo.commit as Any,
            "report_type": isManual ? "manual" : "automated",
        ]
        if let tbStatus = clusterTbBridgeStatus {
            payload["cluster_thunderbolt_bridge"] = tbStatus
        }
        if let desc = userDescription, !desc.isEmpty {
            payload["user_description"] = desc
        }
        return try? JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    }

    /// Extracts cluster-wide Thunderbolt Bridge status from exo state JSON
    private func extractClusterTbBridgeStatus(from stateData: Data?) -> [[String: Any]]? {
        guard let data = stateData,
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let nodeThunderboltBridge = json["node_thunderbolt_bridge"] as? [String: [String: Any]]
        else {
            return nil
        }

        var result: [[String: Any]] = []
        for (nodeId, status) in nodeThunderboltBridge {
            var entry: [String: Any] = ["node_id": nodeId]
            if let enabled = status["enabled"] as? Bool {
                entry["enabled"] = enabled
            }
            if let exists = status["exists"] as? Bool {
                entry["exists"] = exists
            }
            if let serviceName = status["service_name"] as? String {
                entry["service_name"] = serviceName
            }
            result.append(entry)
        }
        return result.isEmpty ? nil : result
    }

    private func readExoMetadata(bundle: Bundle = .main) -> (version: String?, commit: String?) {
        let info = bundle.infoDictionary ?? [:]
        let tag = info["EXOBuildTag"] as? String
        let short = info["CFBundleShortVersionString"] as? String
        let version = [tag, short]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty }
        let commit = (info["EXOBuildCommit"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedCommit = (commit?.isEmpty == true) ? nil : commit
        return (version: version, commit: normalizedCommit)
    }

    private func safeRunCommand(_ arguments: [String]) -> String? {
        let result = runCommand(arguments)
        guard result.exitCode == 0 else { return nil }
        let trimmed = result.output.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func runCommand(_ arguments: [String]) -> CommandResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: arguments[0])
        process.arguments = Array(arguments.dropFirst())

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        do {
            try process.run()
        } catch {
            return CommandResult(exitCode: -1, output: "", error: error.localizedDescription)
        }
        let outputData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        return CommandResult(
            exitCode: process.terminationStatus,
            output: String(decoding: outputData, as: UTF8.self),
            error: String(decoding: errorData, as: UTF8.self)
        )
    }
}

/// The facts about this Mac in a report's report.json, each kept because
/// diagnosing a problem needs it. Not included: serial number, hardware
/// UUID, user and computer names, Wi-Fi network name and MAC addresses.
enum BugReportSystemInfo {
    /// Every field a report can have, and why.
    static let fields: [String: String] = [
        "hardware_model": "Mac model identifier, for chip- and memory-specific bugs",
        "os_version": "macOS version, for bugs specific to one macOS",
        "os_build": "macOS build, to tell betas and updates apart",
        "arch": "arm64, or x86_64 if EXO runs under Rosetta",
        "uptime_seconds": "GPU and memory problems that build up until a restart",
        "default_interface": "which interface reaches other devices and the internet",
        "interfaces": "interface names and addresses, for discovery and connection problems",
    ]

    /// Reads the fields with `run`, which runs a command and returns its
    /// trimmed output, or nil if it fails.
    static func collect(
        run: ([String]) -> String?,
        uptime: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) -> [String: Any] {
        var info: [String: Any] = ["uptime_seconds": Int(uptime)]
        info["hardware_model"] = run(["/usr/sbin/sysctl", "-n", "hw.model"])
        info["os_version"] = run(["/usr/bin/sw_vers", "-productVersion"])
        info["os_build"] = run(["/usr/bin/sw_vers", "-buildVersion"])
        info["arch"] = run(["/usr/bin/uname", "-m"])
        info["default_interface"] = run(["/sbin/route", "-n", "get", "default"])
            .flatMap(parseDefaultInterface)
        info["interfaces"] =
            run(["/usr/sbin/ipconfig", "getiflist"])?
            .split(whereSeparator: { $0 == " " || $0 == "\n" })
            .compactMap { name -> [String: String]? in
                guard let ip = run(["/usr/sbin/ipconfig", "getifaddr", String(name)]) else {
                    return nil
                }
                return ["name": String(name), "ip": ip]
            } ?? []
        return info
    }

    static func parseDefaultInterface(from routeOutput: String) -> String? {
        for line in routeOutput.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("interface: ") {
                return String(trimmed.dropFirst("interface: ".count))
            }
        }
        return nil
    }
}

private struct DebugInfo {
    let thunderboltBridgeDisabled: Bool?
    let interfaces: [InterfaceStatus]
    let rdma: RDMADebugInfo

    struct InterfaceStatus {
        let name: String
        let ip: String?

        func toDictionary() -> [String: Any] {
            [
                "name": name,
                "ip": ip as Any,
            ]
        }
    }

    struct RDMADebugInfo {
        let rdmaCtlStatus: String?
        let ibvDevices: String?
        let ibvDevinfo: String?

        func toDictionary() -> [String: Any] {
            [
                "rdma_ctl_status": rdmaCtlStatus as Any,
                "ibv_devices": ibvDevices as Any,
                "ibv_devinfo": ibvDevinfo as Any,
            ]
        }
    }

    func toDictionary() -> [String: Any] {
        [
            "thunderbolt_bridge_disabled": thunderboltBridgeDisabled as Any,
            "interfaces": interfaces.map { $0.toDictionary() },
            "rdma": rdma.toDictionary(),
        ]
    }
}

private struct CommandResult {
    let exitCode: Int32
    let output: String
    let error: String
}
