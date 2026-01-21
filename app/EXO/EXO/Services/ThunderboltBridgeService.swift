import AppKit
import Combine
import Foundation
import Security
import SystemConfiguration
import os.log

@MainActor
final class ThunderboltBridgeService: ObservableObject {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "ThunderboltBridge")

    @Published private(set) var detectedCycle: [String]?
    @Published private(set) var hasPromptedForCurrentCycle = false
    @Published private(set) var lastError: String?

    private weak var clusterStateService: ClusterStateService?
    private var cancellables = Set<AnyCancellable>()
    private var previousCycleSignature: String?

    init(clusterStateService: ClusterStateService) {
        self.clusterStateService = clusterStateService
        setupObserver()
    }

    private func setupObserver() {
        guard let service = clusterStateService else { return }

        service.$latestSnapshot
            .compactMap { $0 }
            .sink { [weak self] snapshot in
                self?.checkForCycles(snapshot: snapshot)
            }
            .store(in: &cancellables)
    }

    private func checkForCycles(snapshot: ClusterState) {
        let cycles = snapshot.thunderboltBridgeCycles

        // Only consider cycles with more than 2 nodes
        guard let firstCycle = cycles.first, firstCycle.count > 2 else {
            // No problematic cycles detected, reset state
            if detectedCycle != nil {
                detectedCycle = nil
                previousCycleSignature = nil
                hasPromptedForCurrentCycle = false
            }
            return
        }

        // Create a signature for this cycle to detect if it changed
        let cycleSignature = firstCycle.sorted().joined(separator: ",")

        // If this is a new/different cycle, reset the prompt state
        if cycleSignature != previousCycleSignature {
            previousCycleSignature = cycleSignature
            hasPromptedForCurrentCycle = false
        }

        detectedCycle = firstCycle

        // Only prompt once per cycle
        if !hasPromptedForCurrentCycle {
            showDisableBridgePrompt(nodeIds: firstCycle)
        }
    }

    private func showDisableBridgePrompt(nodeIds: [String]) {
        hasPromptedForCurrentCycle = true

        // Get friendly names for the nodes if available
        let nodeNames = nodeIds.map { nodeId -> String in
            if let snapshot = clusterStateService?.latestSnapshot,
                let profile = snapshot.nodeProfiles[nodeId],
                let friendlyName = profile.friendlyName, !friendlyName.isEmpty
            {
                return friendlyName
            }
            return String(nodeId.prefix(8))  // Use first 8 chars of node ID as fallback
        }
        let machineNames = nodeNames.joined(separator: ", ")

        let alert = NSAlert()
        alert.messageText = "Thunderbolt Bridge Loop Detected"
        alert.informativeText = """
            A Thunderbolt Bridge loop has been detected between \(nodeNames.count) machines: \(machineNames).

            This can cause network packet storms and connectivity issues. Would you like to disable Thunderbolt Bridge on this machine to break the loop?
            """
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Disable Bridge")
        alert.addButton(withTitle: "Not Now")

        let response = alert.runModal()

        if response == .alertFirstButtonReturn {
            Task {
                await disableThunderboltBridge()
            }
        }
    }

    func disableThunderboltBridge() async {
        Self.logger.info("Attempting to disable Thunderbolt Bridge via SCPreferences")
        lastError = nil

        do {
            try await disableThunderboltBridgeWithSCPreferences()
            Self.logger.info("Successfully disabled Thunderbolt Bridge")
        } catch {
            Self.logger.error(
                "Failed to disable Thunderbolt Bridge: \(error.localizedDescription, privacy: .public)"
            )
            lastError = error.localizedDescription
            showErrorAlert(message: error.localizedDescription)
        }
    }

    private func disableThunderboltBridgeWithSCPreferences() async throws {
        // 1. Create authorization reference
        var authRef: AuthorizationRef?
        var status = AuthorizationCreate(nil, nil, [], &authRef)
        guard status == errAuthorizationSuccess, let authRef = authRef else {
            throw ThunderboltBridgeError.authorizationFailed
        }

        defer { AuthorizationFree(authRef, [.destroyRights]) }

        // 2. Request specific network configuration rights
        let rightName = "system.services.systemconfiguration.network"
        var item = AuthorizationItem(
            name: rightName,
            valueLength: 0,
            value: nil,
            flags: 0
        )
        var rights = AuthorizationRights(count: 1, items: &item)

        status = AuthorizationCopyRights(
            authRef,
            &rights,
            nil,
            [.extendRights, .interactionAllowed],
            nil
        )
        guard status == errAuthorizationSuccess else {
            if status == errAuthorizationCanceled {
                throw ThunderboltBridgeError.authorizationCanceled
            }
            throw ThunderboltBridgeError.authorizationDenied
        }

        // 3. Create SCPreferences with authorization
        guard
            let prefs = SCPreferencesCreateWithAuthorization(
                kCFAllocatorDefault,
                "EXO" as CFString,
                nil,
                authRef
            )
        else {
            throw ThunderboltBridgeError.preferencesCreationFailed
        }

        // 4. Lock, modify, commit
        guard SCPreferencesLock(prefs, true) else {
            throw ThunderboltBridgeError.lockFailed
        }

        defer {
            SCPreferencesUnlock(prefs)
        }

        // 5. Find the Thunderbolt Bridge service dynamically (don't assume the name)
        guard let targetServiceName = ThunderboltBridgeDetector.findThunderboltBridgeServiceName()
        else {
            throw ThunderboltBridgeError.serviceNotFound
        }

        guard let allServices = SCNetworkServiceCopyAll(prefs) as? [SCNetworkService] else {
            throw ThunderboltBridgeError.servicesNotFound
        }

        var found = false
        for service in allServices {
            if let name = SCNetworkServiceGetName(service) as String?,
                name == targetServiceName
            {
                guard SCNetworkServiceSetEnabled(service, false) else {
                    throw ThunderboltBridgeError.disableFailed
                }
                found = true
                Self.logger.info("Found and disabled Thunderbolt Bridge service: '\(targetServiceName)'")
                break
            }
        }

        if !found {
            throw ThunderboltBridgeError.serviceNotFound
        }

        // 6. Commit and apply
        guard SCPreferencesCommitChanges(prefs) else {
            throw ThunderboltBridgeError.commitFailed
        }

        guard SCPreferencesApplyChanges(prefs) else {
            throw ThunderboltBridgeError.applyFailed
        }
    }

    private func showErrorAlert(message: String) {
        let alert = NSAlert()
        alert.messageText = "Failed to Disable Thunderbolt Bridge"
        alert.informativeText = message
        alert.alertStyle = .critical
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}

enum ThunderboltBridgeError: LocalizedError {
    case authorizationFailed
    case authorizationCanceled
    case authorizationDenied
    case preferencesCreationFailed
    case lockFailed
    case servicesNotFound
    case serviceNotFound
    case disableFailed
    case commitFailed
    case applyFailed

    var errorDescription: String? {
        switch self {
        case .authorizationFailed:
            return "Failed to create authorization"
        case .authorizationCanceled:
            return "Authorization was canceled by user"
        case .authorizationDenied:
            return "Authorization was denied"
        case .preferencesCreationFailed:
            return "Failed to access network preferences"
        case .lockFailed:
            return "Failed to lock network preferences for modification"
        case .servicesNotFound:
            return "Could not retrieve network services"
        case .serviceNotFound:
            return "Thunderbolt Bridge service not found"
        case .disableFailed:
            return "Failed to disable Thunderbolt Bridge service"
        case .commitFailed:
            return "Failed to save network configuration changes"
        case .applyFailed:
            return "Failed to apply network configuration changes"
        }
    }
}
