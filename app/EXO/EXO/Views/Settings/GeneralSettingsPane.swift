import AppKit
import ServiceManagement
import SwiftUI

/// Settings › General: starting EXO, updates, and uninstalling.
struct GeneralSettingsPane: View {
    @EnvironmentObject private var controller: ExoProcessController
    @EnvironmentObject private var updater: SparkleUpdater
    @EnvironmentObject private var stateService: ClusterStateService

    @AppStorage(StartupPreferences.openDashboardOnStartupKey)
    private var openDashboardOnStartup = StartupPreferences.openDashboardOnStartupDefault
    @State private var launchAtLoginStatus: SMAppService.Status = .notRegistered
    @State private var launchAtLoginError: String?
    @State private var uninstallInProgress = false

    var body: some View {
        Form {
            Section {
                Toggle("Open EXO at login", isOn: launchAtLoginBinding)
                if let launchAtLoginError {
                    Label(launchAtLoginError, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.secondary)
                } else if launchAtLoginStatus == .requiresApproval {
                    LabeledContent {
                        Button("Open Login Items…") {
                            SMAppService.openSystemSettingsLoginItems()
                        }
                    } label: {
                        Label(
                            "EXO needs your approval in Login Items to open at login.",
                            systemImage: "exclamationmark.triangle")
                    }
                }
                Toggle("Open dashboard when EXO starts", isOn: $openDashboardOnStartup)
            } header: {
                Text("Startup")
            } footer: {
                SettingFootnote(
                    "When you open EXO yourself. Nothing opens when EXO starts at login.")
            }

            Section("Software Update") {
                Toggle(
                    "Check for updates automatically",
                    isOn: $updater.automaticallyChecksForUpdates)
                Toggle(
                    "Download and install updates automatically",
                    isOn: $updater.automaticallyDownloadsUpdates
                )
                .disabled(!updater.automaticallyChecksForUpdates)
                LabeledContent("EXO \(ExoProcessController.buildTag)") {
                    Button("Check Now") {
                        updater.checkForUpdates()
                    }
                }
            }

            Section {
                HStack {
                    Button("Uninstall EXO…") {
                        uninstallInProgress = true
                        Task {
                            await UninstallEXO.run(
                                controller: controller, stateService: stateService)
                            uninstallInProgress = false
                        }
                    }
                    .disabled(uninstallInProgress)
                    Spacer()
                }
            }
        }
        .settingsPane()
        .onAppear { launchAtLoginStatus = LaunchAtLoginHelper.status }
        .onReceive(
            NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)
        ) { _ in
            // Picks up changes made in System Settings › Login Items.
            launchAtLoginStatus = LaunchAtLoginHelper.status
        }
    }

    /// Reflects the login item's actual state in System Settings rather than a
    /// stored copy, so it stays right if the user changes it there.
    private var launchAtLoginBinding: Binding<Bool> {
        Binding(
            get: {
                launchAtLoginStatus == .enabled || launchAtLoginStatus == .requiresApproval
            },
            set: { enabled in
                // The user has decided, so startup no longer changes it.
                StartupPreferences().markLaunchAtLoginDefaultApplied()
                do {
                    try LaunchAtLoginHelper.setEnabled(enabled)
                    launchAtLoginError = nil
                } catch {
                    launchAtLoginError =
                        "EXO couldn't \(enabled ? "turn on" : "turn off") opening at login: "
                        + error.localizedDescription
                }
                launchAtLoginStatus = LaunchAtLoginHelper.status
            }
        )
    }
}
