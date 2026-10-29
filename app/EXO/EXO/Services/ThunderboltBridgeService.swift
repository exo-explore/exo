import AppKit
import Combine
import Foundation
import Security
import SystemConfiguration
import os.log

@MainActor
final class ThunderboltBridgeService: ObservableObject {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "ThunderboltBridge")

    /// Devices whose Thunderbolt Bridges form a loop (more than two), as the
    /// cluster reports it. Shown as a row in the menu and a badge on the
    /// icon, rather than an alert at an arbitrary moment.
    @Published private(set) var detectedCycle: [String]?
    @Published private(set) var lastError: String?

    private weak var clusterStateService: ClusterStateService?
    private var cancellables = Set<AnyCancellable>()

    init(clusterStateService: ClusterStateService) {
        self.clusterStateService = clusterStateService
        setupObserver()
    }

    private func setupObserver() {
        guard let service = clusterStateService else { return }

        service.$latestSnapshot
            .map { snapshot in Self.loop(in: snapshot?.thunderboltBridgeCycles ?? []) }
            .removeDuplicates()
            .sink { [weak self] cycle in
                self?.detectedCycle = cycle
            }
            .store(in: &cancellables)
    }

    /// The first loop of more than two devices; a pair can't loop.
    static func loop(in cycles: [[String]]) -> [String]? {
        cycles.first { $0.count > 2 }
    }

    /// Whether this Mac is in the loop, so turning off its bridge helps.
    func loopIncludes(nodeId: String?) -> Bool {
        guard let nodeId, let detectedCycle else { return false }
        return detectedCycle.contains(nodeId)
    }

    func disableThunderboltBridge() async {
        Self.logger.info("Attempting to disable Thunderbolt Bridge via SCPreferences")
        lastError = nil

        do {
            try await disableThunderboltBridgeWithSCPreferences()
            Self.logger.info("Successfully disabled Thunderbolt Bridge")
        } catch ThunderboltBridgeError.authorizationCanceled {
            // The user cancelled the password prompt: their choice, not an error.
            Self.logger.info("Turning off Thunderbolt Bridge was cancelled")
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
        status = rightName.withCString { nameCString in
            var item = AuthorizationItem(
                name: nameCString,
                valueLength: 0,
                value: nil,
                flags: 0
            )
            return withUnsafeMutablePointer(to: &item) { itemPointer in
                var rights = AuthorizationRights(count: 1, items: itemPointer)
                return AuthorizationCopyRights(
                    authRef,
                    &rights,
                    nil,
                    [.extendRights, .interactionAllowed],
                    nil
                )
            }
        }
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
                Self.logger.info(
                    "Found and disabled Thunderbolt Bridge service: '\(targetServiceName)'")
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
        alert.messageText = "Couldn't Turn Off Thunderbolt Bridge"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModalInFront()
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
