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

struct BugReportService {
    private struct PresignedUrlsRequest: Codable {
        let keys: [String]
    }

    private struct PresignedUrlsResponse: Codable {
        let urls: [String: String]
        let expiresIn: Int?
    }

    func sendReport(
        baseURL: URL = URL(string: "http://127.0.0.1:52415")!,
        now: Date = Date(),
        isManual: Bool = false
    ) async throws -> BugReportOutcome {
        let timestamp = Self.runTimestampString(now)
        let dayPrefix = Self.dayPrefixString(now)
        let prefix = "reports/\(dayPrefix)/\(timestamp)/"

        let logData = readLog()
        let ifconfigText = try await captureIfconfig()
        let hostName = Host.current().localizedName ?? "unknown"
        let debugInfo = readDebugInfo()

        async let stateResult = fetch(url: baseURL.appendingPathComponent("state"))
        async let eventsResult = fetch(url: baseURL.appendingPathComponent("events"))

        let stateData = try await stateResult
        let eventsData = try await eventsResult

        let reportJSON = makeReportJson(
            timestamp: timestamp,
            hostName: hostName,
            ifconfig: ifconfigText,
            debugInfo: debugInfo,
            isManual: isManual
        )

        let uploads: [(path: String, data: Data?)] = [
            ("\(prefix)exo.log", logData),
            ("\(prefix)state.json", stateData),
            ("\(prefix)events.json", eventsData),
            ("\(prefix)report.json", reportJSON),
        ]

        let uploadItems: [(key: String, body: Data)] = uploads.compactMap { item in
            guard let body = item.data else { return nil }
            return (key: item.path, body: body)
        }

        guard !uploadItems.isEmpty else {
            return BugReportOutcome(success: false, message: "No data to upload")
        }

        let presignedUrls = try await fetchPresignedUploadUrls(keys: uploadItems.map(\.key))
        for item in uploadItems {
            guard let urlString = presignedUrls[item.key], let url = URL(string: urlString) else {
                throw BugReportError.uploadFailed("Missing presigned URL for \(item.key)")
            }
            try await uploadToPresignedUrl(url: url, body: item.body)
        }

        return BugReportOutcome(
            success: true, message: "Bug Report sent. Thank you for helping to improve EXO 1.0.")
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

    private static func runTimestampString(_ date: Date) -> String {
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

    private func readLog() -> Data? {
        let logURL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".exo")
            .appendingPathComponent("exo.log")
        return try? Data(contentsOf: logURL)
    }

    private func captureIfconfig() async throws -> String {
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
            ibvDevices: safeRunCommand(["/usr/bin/ibv_devices"]),
            ibvDevinfo: safeRunCommand(["/usr/bin/ibv_devinfo"])
        )
    }

    private func readThunderboltBridgeDisabled() -> Bool? {
        let result = runCommand([
            "/usr/sbin/networksetup", "-getnetworkserviceenabled", "Thunderbolt Bridge",
        ])
        guard result.exitCode == 0 else { return nil }
        let output = result.output.lowercased()
        if output.contains("enabled") {
            return false
        }
        if output.contains("disabled") {
            return true
        }
        return nil
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

    private func fetch(url: URL) async throws -> Data? {
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
        hostName: String,
        ifconfig: String,
        debugInfo: DebugInfo,
        isManual: Bool
    ) -> Data? {
        let system = readSystemMetadata()
        let exo = readExoMetadata()
        let payload: [String: Any] = [
            "timestamp": timestamp,
            "host": hostName,
            "ifconfig": ifconfig,
            "debug": debugInfo.toDictionary(),
            "system": system,
            "exo_version": exo.version as Any,
            "exo_commit": exo.commit as Any,
            "report_type": isManual ? "manual" : "automated",
        ]
        return try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
    }

    private func readSystemMetadata() -> [String: Any] {
        let hostname = safeRunCommand(["/bin/hostname"])
        let computerName = safeRunCommand(["/usr/sbin/scutil", "--get", "ComputerName"])
        let localHostName = safeRunCommand(["/usr/sbin/scutil", "--get", "LocalHostName"])
        let hostNameCommand = safeRunCommand(["/usr/sbin/scutil", "--get", "HostName"])
        let hardwareModel = safeRunCommand(["/usr/sbin/sysctl", "-n", "hw.model"])
        let hardwareProfile = safeRunCommand(["/usr/sbin/system_profiler", "SPHardwareDataType"])
        let hardwareUUID = hardwareProfile.flatMap(extractHardwareUUID)

        let osVersion = safeRunCommand(["/usr/bin/sw_vers", "-productVersion"])
        let osBuild = safeRunCommand(["/usr/bin/sw_vers", "-buildVersion"])
        let kernel = safeRunCommand(["/usr/bin/uname", "-srv"])
        let arch = safeRunCommand(["/usr/bin/uname", "-m"])

        let routeInfo = safeRunCommand(["/sbin/route", "-n", "get", "default"])
        let defaultInterface = routeInfo.flatMap(parseDefaultInterface)
        let defaultIP = defaultInterface.flatMap { iface in
            safeRunCommand(["/usr/sbin/ipconfig", "getifaddr", iface])
        }
        let defaultMac = defaultInterface.flatMap { iface in
            safeRunCommand(["/sbin/ifconfig", iface]).flatMap(parseEtherAddress)
        }

        let user = safeRunCommand(["/usr/bin/whoami"])
        let consoleUser = safeRunCommand(["/usr/bin/stat", "-f%Su", "/dev/console"])
        let uptime = safeRunCommand(["/usr/bin/uptime"])
        let diskRoot = safeRunCommand([
            "/bin/sh", "-c", "/bin/df -h / | awk 'NR==2 {print $1, $2, $3, $4, $5}'",
        ])

        let interfacesList = safeRunCommand(["/usr/sbin/ipconfig", "getiflist"])
        let interfacesAndIPs =
            interfacesList?
            .split(whereSeparator: { $0 == " " || $0 == "\n" })
            .compactMap { iface -> [String: Any]? in
                let name = String(iface)
                guard let ip = safeRunCommand(["/usr/sbin/ipconfig", "getifaddr", name]) else {
                    return nil
                }
                return ["name": name, "ip": ip]
            } ?? []

        let wifiSSID: String?
        let airportPath =
            "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
        if FileManager.default.isExecutableFile(atPath: airportPath) {
            wifiSSID = safeRunCommand([airportPath, "-I"]).flatMap(parseWifiSSID)
        } else {
            wifiSSID = nil
        }

        return [
            "hostname": hostname as Any,
            "computer_name": computerName as Any,
            "local_hostname": localHostName as Any,
            "host_name": hostNameCommand as Any,
            "hardware_model": hardwareModel as Any,
            "hardware_profile": hardwareProfile as Any,
            "hardware_uuid": hardwareUUID as Any,
            "os_version": osVersion as Any,
            "os_build": osBuild as Any,
            "kernel": kernel as Any,
            "arch": arch as Any,
            "default_interface": defaultInterface as Any,
            "default_ip": defaultIP as Any,
            "default_mac": defaultMac as Any,
            "user": user as Any,
            "console_user": consoleUser as Any,
            "uptime": uptime as Any,
            "disk_root": diskRoot as Any,
            "interfaces_and_ips": interfacesAndIPs,
            "ipconfig_getiflist": interfacesList as Any,
            "wifi_ssid": wifiSSID as Any,
        ]
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

    private func extractHardwareUUID(from hardwareProfile: String) -> String? {
        hardwareProfile
            .split(separator: "\n")
            .first { $0.contains("Hardware UUID") }?
            .split(separator: ":")
            .dropFirst()
            .joined(separator: ":")
            .trimmingCharacters(in: .whitespaces)
    }

    private func parseDefaultInterface(from routeOutput: String) -> String? {
        for line in routeOutput.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("interface: ") {
                return trimmed.replacingOccurrences(of: "interface: ", with: "")
            }
        }
        return nil
    }

    private func parseEtherAddress(from ifconfigOutput: String) -> String? {
        for line in ifconfigOutput.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("ether ") {
                return trimmed.replacingOccurrences(of: "ether ", with: "")
            }
        }
        return nil
    }

    private func parseWifiSSID(from airportOutput: String) -> String? {
        for line in airportOutput.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("SSID:") {
                return trimmed.replacingOccurrences(of: "SSID:", with: "").trimmingCharacters(
                    in: .whitespaces)
            }
        }
        return nil
    }

    private func runCommand(_ arguments: [String]) -> CommandResult {
        let process = Process()
        process.launchPath = arguments.first
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
        process.waitUntilExit()

        let outputData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errorData = stderr.fileHandleForReading.readDataToEndOfFile()

        return CommandResult(
            exitCode: process.terminationStatus,
            output: String(decoding: outputData, as: UTF8.self),
            error: String(decoding: errorData, as: UTF8.self)
        )
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
