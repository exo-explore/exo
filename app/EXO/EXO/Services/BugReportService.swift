import CryptoKit
import Foundation

struct BugReportOutcome: Equatable {
    let success: Bool
    let message: String
}

enum BugReportError: LocalizedError {
    case missingCredentials
    case invalidEndpoint
    case uploadFailed(String)
    case collectFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingCredentials:
            return "Bug report upload credentials are not set."
        case .invalidEndpoint:
            return "Bug report endpoint is invalid."
        case .uploadFailed(let message):
            return "Bug report upload failed: \(message)"
        case .collectFailed(let message):
            return "Bug report collection failed: \(message)"
        }
    }
}

struct BugReportService {
    struct AWSConfig {
        let accessKey: String
        let secretKey: String
        let region: String
        let bucket: String
    }

    func sendReport(
        baseURL: URL = URL(string: "http://127.0.0.1:52415")!,
        now: Date = Date(),
        isManual: Bool = false
    ) async throws -> BugReportOutcome {
        let credentials = try loadCredentials()
        let timestamp = ISO8601DateFormatter().string(from: now)
        let prefix = "reports/\(timestamp)/"

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

        let uploader = try S3Uploader(config: credentials)
        for item in uploads {
            guard let data = item.data else { continue }
            try await uploader.upload(
                objectPath: item.path,
                body: data
            )
        }

        return BugReportOutcome(
            success: true, message: "Bug Report sent. Thank you for helping to improve EXO 1.0.")
    }

    private func loadCredentials() throws -> AWSConfig {
        return AWSConfig(
            accessKey: "AKIAYEKP5EMXTOBYDGHX",
            secretKey: "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",
            region: "us-east-1",
            bucket: "exo-bug-reports"
        )
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

private struct S3Uploader {
    let config: BugReportService.AWSConfig

    init(config: BugReportService.AWSConfig) throws {
        self.config = config
    }

    func upload(objectPath: String, body: Data) async throws {
        let host = "\(config.bucket).s3.amazonaws.com"
        guard let url = URL(string: "https://\(host)/\(objectPath)") else {
            throw BugReportError.invalidEndpoint
        }

        let now = Date()
        let amzDate = awsTimestamp(now)
        let dateStamp = dateStamp(now)
        let payloadHash = sha256Hex(body)

        let headers = [
            "host": host,
            "x-amz-content-sha256": payloadHash,
            "x-amz-date": amzDate,
        ]

        let canonicalRequest = buildCanonicalRequest(
            method: "PUT",
            url: url,
            headers: headers,
            payloadHash: payloadHash
        )

        let stringToSign = buildStringToSign(
            amzDate: amzDate,
            dateStamp: dateStamp,
            canonicalRequestHash: sha256Hex(canonicalRequest.data(using: .utf8) ?? Data())
        )

        let signingKey = deriveKey(
            secret: config.secretKey, dateStamp: dateStamp, region: config.region, service: "s3")
        let signature = hmacHex(key: signingKey, data: Data(stringToSign.utf8))

        let signedHeaders = "host;x-amz-content-sha256;x-amz-date"
        let authorization = """
            AWS4-HMAC-SHA256 Credential=\(config.accessKey)/\(dateStamp)/\(config.region)/s3/aws4_request, SignedHeaders=\(signedHeaders), Signature=\(signature)
            """

        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.httpBody = body
        request.setValue(
            headers["x-amz-content-sha256"], forHTTPHeaderField: "x-amz-content-sha256")
        request.setValue(headers["x-amz-date"], forHTTPHeaderField: "x-amz-date")
        request.setValue(host, forHTTPHeaderField: "Host")
        request.setValue(authorization, forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let statusText = (response as? HTTPURLResponse)?.statusCode ?? -1
            _ = data  // ignore response body for UX
            throw BugReportError.uploadFailed("HTTP status \(statusText)")
        }
    }

    private func buildCanonicalRequest(
        method: String,
        url: URL,
        headers: [String: String],
        payloadHash: String
    ) -> String {
        let canonicalURI = encodePath(url.path)
        let canonicalQuery = url.query ?? ""
        let sortedHeaders = headers.sorted { $0.key < $1.key }
        let canonicalHeaders =
            sortedHeaders
            .map { "\($0.key.lowercased()):\($0.value)\n" }
            .joined()
        let signedHeaders = sortedHeaders.map { $0.key.lowercased() }.joined(separator: ";")

        return [
            method,
            canonicalURI,
            canonicalQuery,
            canonicalHeaders,
            signedHeaders,
            payloadHash,
        ].joined(separator: "\n")
    }

    private func encodePath(_ path: String) -> String {
        return
            path
            .split(separator: "/")
            .map { segment in
                segment.addingPercentEncoding(withAllowedCharacters: Self.rfc3986)
                    ?? String(segment)
            }
            .joined(separator: "/")
            .prependSlashIfNeeded()
    }

    private func buildStringToSign(
        amzDate: String,
        dateStamp: String,
        canonicalRequestHash: String
    ) -> String {
        """
        AWS4-HMAC-SHA256
        \(amzDate)
        \(dateStamp)/\(config.region)/s3/aws4_request
        \(canonicalRequestHash)
        """
    }

    private func deriveKey(secret: String, dateStamp: String, region: String, service: String)
        -> Data
    {
        let kDate = hmac(key: Data(("AWS4" + secret).utf8), data: Data(dateStamp.utf8))
        let kRegion = hmac(key: kDate, data: Data(region.utf8))
        let kService = hmac(key: kRegion, data: Data(service.utf8))
        return hmac(key: kService, data: Data("aws4_request".utf8))
    }

    private func hmac(key: Data, data: Data) -> Data {
        let keySym = SymmetricKey(data: key)
        let mac = HMAC<SHA256>.authenticationCode(for: data, using: keySym)
        return Data(mac)
    }

    private func hmacHex(key: Data, data: Data) -> String {
        hmac(key: key, data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func sha256Hex(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.compactMap { String(format: "%02x", $0) }.joined()
    }

    private func awsTimestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        formatter.timeZone = TimeZone(abbreviation: "UTC")
        return formatter.string(from: date)
    }

    private func dateStamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd"
        formatter.timeZone = TimeZone(abbreviation: "UTC")
        return formatter.string(from: date)
    }

    private static let rfc3986: CharacterSet = {
        var set = CharacterSet.alphanumerics
        set.insert(charactersIn: "-._~")
        return set
    }()
}

extension String {
    fileprivate func prependSlashIfNeeded() -> String {
        if hasPrefix("/") {
            return self
        }
        return "/" + self
    }
}
