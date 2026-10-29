import AppKit
import os.log

/// Settings › General › Uninstall EXO….
@MainActor
enum UninstallEXO {
    private static let logger = Logger(subsystem: "io.exo.EXO", category: "Uninstall")

    /// Asks, then removes EXO's network service, its data in ~/.exo (keeping
    /// downloaded models unless the user unticks that), its login item, its
    /// Hugging Face token and its preferences, moves the app to the Trash and
    /// quits.
    ///
    /// The network service is removed first, while exo keeps running: it is
    /// the step that needs an administrator password, so cancelling the
    /// password prompt leaves EXO exactly as it was, with nothing to report.
    static func run(controller: ExoProcessController, stateService: ClusterStateService) async {
        let (alert, keepModels) = makeAlert(modelsSize: downloadedModelsSize())
        guard alert.runModalInFront() == .alertFirstButtonReturn else { return }
        let keepsModels = keepModels.state == .on

        let removal = await Task.detached(priority: .utility) {
            Result { try NetworkSetupHelper.uninstall() }
        }.value
        if case .failure(let error) = removal {
            if error as? NetworkSetupError == .cancelled {
                logger.info("Uninstall cancelled at the password prompt")
            } else {
                showFailure(error)
            }
            SettingsWindowController.bringSettingsToFront()
            return
        }

        controller.cancelPendingLaunch()
        stateService.stopPolling()
        await controller.stopAndWaitForExit()
        do {
            try removeExoDirectory(keepModels: keepsModels)
        } catch {
            logger.error(
                "Couldn't remove all of ~/.exo: \(error.localizedDescription, privacy: .public)")
        }
        do {
            try HuggingFaceToken.keychain.delete()
        } catch {
            logger.error(
                "Couldn't delete the Hugging Face token: \(error.localizedDescription, privacy: .public)"
            )
        }
        LaunchAtLoginHelper.disable()
        removePreferences()
        moveAppToTrash()
        quitSoon()
    }

    /// Quits from the run loop rather than from this task. Quitting waits
    /// for exo to stop (`AppDelegate.applicationShouldTerminate`) in another
    /// main-actor task, which can't run while this one is still inside
    /// `terminate`.
    private static func quitSoon() {
        NSApp.perform(#selector(NSApplication.terminate(_:)), with: nil, afterDelay: 0.5)
    }

    /// The Uninstall confirmation. "Keep downloaded models" starts ticked:
    /// models can take hundreds of gigabytes to download again, so deleting
    /// them has to be chosen.
    static func makeAlert(modelsSize: Int64?) -> (alert: NSAlert, keepModels: NSButton) {
        let alert = NSAlert()
        alert.messageText = "Uninstall EXO?"
        alert.informativeText =
            "EXO, its network service and its settings will be removed, and EXO will be moved "
            + "to the Trash."
        alert.alertStyle = .warning

        var keepTitle = "Keep downloaded models"
        if let modelsSize, modelsSize > 0 {
            keepTitle +=
                " (\(ByteCountFormatter.string(fromByteCount: modelsSize, countStyle: .file)))"
        }
        let checkbox = NSButton(checkboxWithTitle: keepTitle, target: nil, action: nil)
        checkbox.state = .on
        checkbox.sizeToFit()
        alert.accessoryView = checkbox

        alert.addButton(withTitle: "Uninstall")
        alert.addButton(withTitle: "Cancel")
        if let uninstallButton = alert.buttons.first {
            uninstallButton.hasDestructiveAction = true
        }
        return (alert, checkbox)
    }

    private static func showFailure(_ error: Error) {
        let alert = NSAlert()
        alert.messageText = "EXO Wasn't Uninstalled"
        alert.informativeText = error.localizedDescription
        alert.alertStyle = .warning
        alert.runModalInFront()
    }

    private static var modelsDirectory: URL {
        ExoProcessController.exoDirectoryURL.appendingPathComponent("models", isDirectory: true)
    }

    /// The disk space the models in ~/.exo/models take, or nil if there are
    /// none.
    private static func downloadedModelsSize() -> Int64? {
        guard
            let enumerator = FileManager.default.enumerator(
                at: modelsDirectory,
                includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .isRegularFileKey])
        else { return nil }
        var total: Int64 = 0
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [
                .totalFileAllocatedSizeKey, .isRegularFileKey,
            ])
            if values?.isRegularFile == true {
                total += Int64(values?.totalFileAllocatedSize ?? 0)
            }
        }
        return total > 0 ? total : nil
    }

    private static func removeExoDirectory(keepModels: Bool) throws {
        let fileManager = FileManager.default
        let exoDirectory = ExoProcessController.exoDirectoryURL
        guard fileManager.fileExists(atPath: exoDirectory.path) else { return }

        if !keepModels {
            try fileManager.removeItem(at: exoDirectory)
            return
        }

        let contents = try fileManager.contentsOfDirectory(
            at: exoDirectory, includingPropertiesForKeys: nil, options: [])
        for entry in contents where entry.lastPathComponent != "models" {
            try? fileManager.removeItem(at: entry)
        }
    }

    /// Removes all of EXO's preferences, so that installing it again starts
    /// afresh. Windows are closed first, because a window saves its frame
    /// there when it closes, and the removal is repeated as the app quits, in
    /// case anything saved one since.
    private static func removePreferences() {
        guard let domain = Bundle.main.bundleIdentifier else { return }
        for window in NSApp.windows where window.styleMask.contains(.titled) {
            window.close()
        }
        UserDefaults.standard.removePersistentDomain(forName: domain)
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { _ in
            UserDefaults.standard.removePersistentDomain(forName: domain)
        }
    }

    private static func moveAppToTrash() {
        do {
            try FileManager.default.trashItem(at: Bundle.main.bundleURL, resultingItemURL: nil)
        } catch {
            // The user can still drag it to the Trash.
            logger.error(
                "Couldn't move EXO to the Trash: \(error.localizedDescription, privacy: .public)")
        }
    }
}
