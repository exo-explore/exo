import Foundation
import os
import UIKit

struct DeviceInfo: Encodable {
    let nodeId: String
    let model: String
    let chip: String
    let osVersion: String
    let friendlyName: String
    let ramTotal: Int
    let ramAvailable: Int

    enum CodingKeys: String, CodingKey {
        case nodeId = "node_id"
        case model
        case chip
        case osVersion = "os_version"
        case friendlyName = "friendly_name"
        case ramTotal = "ram_total"
        case ramAvailable = "ram_available"
    }
}

enum DeviceInfoService {
    private static let liteNodeIdKey = "exo_lite_node_id"

    static var liteNodeId: String {
        if let existing = UserDefaults.standard.string(forKey: liteNodeIdKey) {
            return existing
        }
        let newId = UUID().uuidString.lowercased()
        UserDefaults.standard.set(newId, forKey: liteNodeIdKey)
        return newId
    }

    static func gather() -> DeviceInfo {
        let model = modelName()
        let chip = chipName(for: model)

        let totalRam = Int(ProcessInfo.processInfo.physicalMemory)
        let availableRam = availableMemory()

        return DeviceInfo(
            nodeId: liteNodeId,
            model: model,
            chip: chip,
            osVersion: UIDevice.current.systemVersion,
            friendlyName: UIDevice.current.name,
            ramTotal: totalRam,
            ramAvailable: availableRam
        )
    }

    // MARK: - Private

    private static func modelName() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machine = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(cString: $0)
            }
        }
        return modelMapping[machine] ?? machine
    }

    private static func chipName(for model: String) -> String {
        let lower = model.lowercased()
        if lower.contains("iphone 16 pro") || lower.contains("iphone 16 pro max") {
            return "Apple A18 Pro"
        } else if lower.contains("iphone 16") {
            return "Apple A18"
        } else if lower.contains("iphone 15 pro") || lower.contains("iphone 15 pro max") {
            return "Apple A17 Pro"
        } else if lower.contains("iphone 15") {
            return "Apple A16 Bionic"
        } else if lower.contains("iphone 14 pro") || lower.contains("iphone 14 pro max") {
            return "Apple A16 Bionic"
        } else if lower.contains("iphone 14") {
            return "Apple A15 Bionic"
        }
        return "Apple Silicon"
    }

    private static func availableMemory() -> Int {
        return Int(os_proc_available_memory())
    }

    private static let modelMapping: [String: String] = [
        // iPhone 16 series
        "iPhone17,1": "iPhone 16 Pro",
        "iPhone17,2": "iPhone 16 Pro Max",
        "iPhone17,3": "iPhone 16",
        "iPhone17,4": "iPhone 16 Plus",
        // iPhone 15 series
        "iPhone16,1": "iPhone 15 Pro",
        "iPhone16,2": "iPhone 15 Pro Max",
        "iPhone15,4": "iPhone 15",
        "iPhone15,5": "iPhone 15 Plus",
        // iPhone 14 series
        "iPhone15,2": "iPhone 14 Pro",
        "iPhone15,3": "iPhone 14 Pro Max",
        "iPhone14,7": "iPhone 14",
        "iPhone14,8": "iPhone 14 Plus",
        // iPhone 13 series
        "iPhone14,2": "iPhone 13 Pro",
        "iPhone14,3": "iPhone 13 Pro Max",
        "iPhone14,5": "iPhone 13",
        "iPhone14,4": "iPhone 13 mini",
        // Simulator
        "arm64": "iPhone (Simulator)",
        "x86_64": "iPhone (Simulator)",
    ]
}
