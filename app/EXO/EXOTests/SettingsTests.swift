import AppKit
import Foundation
import Testing

@testable import EXO

/// Throwaway defaults in a temporary plist, so tests never touch the app's
/// real preferences.
final class ScratchDefaults {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("EXOSettingsTests-\(UUID().uuidString)", isDirectory: true)
    let suiteName: String
    let defaults: UserDefaults

    init() {
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        suiteName = directory.appendingPathComponent("defaults").path
        defaults = UserDefaults(suiteName: suiteName)!
    }

    deinit {
        defaults.removePersistentDomain(forName: suiteName)
        try? FileManager.default.removeItem(at: directory)
    }
}

// MARK: - Fast synch

struct FastSynchSettingTests {

    @Test func aNewInstallIsAutomatic() {
        let storage = ScratchDefaults()
        #expect(FastSynchSetting.load(from: storage.defaults) == .automatic)
        #expect(storage.defaults.string(forKey: FastSynchSetting.storageKey) == "automatic")
    }

    @Test func theOldDefaultOfOnBecomesAutomatic() {
        // Earlier versions defaulted the toggle to on, which forced fast synch
        // for every model; saving any setting stored that default.
        let storage = ScratchDefaults()
        storage.defaults.set(true, forKey: FastSynchSetting.legacyEnabledKey)

        #expect(FastSynchSetting.load(from: storage.defaults) == .automatic)
        #expect(storage.defaults.object(forKey: FastSynchSetting.legacyEnabledKey) == nil)
    }

    @Test func turningItOffIsKept() {
        let storage = ScratchDefaults()
        storage.defaults.set(false, forKey: FastSynchSetting.legacyEnabledKey)

        #expect(FastSynchSetting.load(from: storage.defaults) == .off)
        // Converted once: the next launch reads the new key.
        #expect(storage.defaults.string(forKey: FastSynchSetting.storageKey) == "off")
        #expect(FastSynchSetting.load(from: storage.defaults) == .off)
    }

    @Test func aStoredChoiceWinsOverTheOldToggle() {
        let storage = ScratchDefaults()
        storage.defaults.set("on", forKey: FastSynchSetting.storageKey)
        storage.defaults.set(false, forKey: FastSynchSetting.legacyEnabledKey)

        #expect(FastSynchSetting.load(from: storage.defaults) == .on)
    }

    @Test func automaticLeavesTheChoiceToExo() {
        #expect(FastSynchSetting.automatic.environmentValue == nil)
        #expect(FastSynchSetting.on.environmentValue == "true")
        #expect(FastSynchSetting.off.environmentValue == "false")
    }
}

// MARK: - Model folders

struct ModelFolderTests {

    @Test func theTwoOldListsBecomeOneListWithAReadOnlyColumn() {
        let storage = ScratchDefaults()
        storage.defaults.set(
            "/Volumes/Fast/models: /Users/shared/models", forKey: ModelFolder.legacyWritableKey)
        storage.defaults.set("/Volumes/Archive/models", forKey: ModelFolder.legacyReadOnlyKey)

        let folders = ModelFolder.load(from: storage.defaults)

        #expect(
            folders.map(\.path) == [
                "/Volumes/Fast/models", "/Users/shared/models", "/Volumes/Archive/models",
            ])
        #expect(folders.map(\.readOnly) == [false, false, true])
        #expect(storage.defaults.object(forKey: ModelFolder.legacyWritableKey) == nil)
        #expect(storage.defaults.object(forKey: ModelFolder.legacyReadOnlyKey) == nil)
        // Saved in the new form, so it is read back as it is.
        #expect(ModelFolder.load(from: storage.defaults) == folders)
    }

    @Test func aFolderInBothOldListsIsReadOnly() {
        // That is how exo treated it.
        let storage = ScratchDefaults()
        storage.defaults.set("/a:/b", forKey: ModelFolder.legacyWritableKey)
        storage.defaults.set("/b", forKey: ModelFolder.legacyReadOnlyKey)

        let folders = ModelFolder.load(from: storage.defaults)

        #expect(folders.map(\.path) == ["/a", "/b"])
        #expect(folders.map(\.readOnly) == [false, true])
    }

    @Test func foldersBecomeExosTwoColonSeparatedLists() {
        let environment = ModelFolder.environment(for: [
            ModelFolder(path: "/a"),
            ModelFolder(path: " /b ", readOnly: true),
            ModelFolder(path: "/c"),
        ])
        #expect(environment == ["EXO_MODELS_DIRS": "/a:/c", "EXO_MODELS_READ_ONLY_DIRS": "/b"])
    }

    @Test func aFolderExoCantTakeIsLeftOut() {
        // exo splits the lists on colons.
        let folders = [ModelFolder(path: "/odd:name"), ModelFolder(path: "  ")]
        #expect(folders.allSatisfy { !$0.isUsable })
        #expect(ModelFolder.environment(for: folders).isEmpty)
    }
}

// MARK: - exo's environment

struct ExoLaunchSettingsTests {

    @Test func defaultsGiveOnlyTheVersionAsClusterName() {
        #expect(
            ExoLaunchSettings().environment(buildTag: "1.0.71") == ["EXO_ZENOH_NAMESPACE": "1.0.71"]
        )
    }

    @Test func eachSettingReachesExo() {
        var settings = ExoLaunchSettings()
        settings.clusterName = " studio "
        settings.huggingFaceToken = "hf_secret"
        settings.huggingFaceEndpoint = "hf-mirror.com/"
        settings.showImageModels = true
        settings.offline = true
        settings.fastSynch = .off
        settings.downloadFolder = "/Volumes/Fast/models"
        settings.otherModelFolders = [ModelFolder(path: "/Volumes/Archive", readOnly: true)]
        settings.customEnvironmentVariables = [
            CustomEnvironmentVariable(key: "MY_FLAG", value: "1")
        ]

        #expect(
            settings.environment(buildTag: "1.0.71") == [
                "EXO_ZENOH_NAMESPACE": "studio",
                "HF_TOKEN": "hf_secret",
                "HF_ENDPOINT": "https://hf-mirror.com",
                "EXO_ENABLE_IMAGE_MODELS": "true",
                "EXO_OFFLINE": "true",
                "EXO_FAST_SYNCH": "false",
                "EXO_DEFAULT_MODELS_DIR": "/Volumes/Fast/models",
                "EXO_MODELS_READ_ONLY_DIRS": "/Volumes/Archive",
                "MY_FLAG": "1",
            ])
    }

    @Test func fastSynchOnIsPassed() {
        var settings = ExoLaunchSettings()
        settings.fastSynch = .on
        #expect(settings.environment(buildTag: "t")["EXO_FAST_SYNCH"] == "true")
    }

    @Test func customVariablesOverrideTheOtherSettingsAndInvalidOnesAreLeftOut() {
        var settings = ExoLaunchSettings()
        settings.offline = true
        settings.customEnvironmentVariables = [
            CustomEnvironmentVariable(key: "EXO_OFFLINE", value: "false"),
            CustomEnvironmentVariable(key: "1BAD", value: "x"),
        ]
        let environment = settings.environment(buildTag: "t")
        #expect(environment["EXO_OFFLINE"] == "false")
        #expect(environment["1BAD"] == nil)
    }

    @Test func exosEnvironmentPutsTheRuntimeFirstOnPathAndSettingsLast() {
        let runtime = URL(fileURLWithPath: "/Applications/EXO.app/Contents/Resources/exo")
        let environment = ExoProcessController.makeEnvironment(
            base: ["PATH": "/usr/bin:/custom/bin", "HOME": "/Users/test", "EXO_OFFLINE": "true"],
            runtimeURL: runtime,
            settingsEnvironment: ["EXO_ZENOH_NAMESPACE": "studio", "PATH": "/override"])

        #expect(environment["HOME"] == "/Users/test")
        #expect(environment["EXO_RUNTIME_DIR"] == runtime.path)
        #expect(environment["EXO_ZENOH_NAMESPACE"] == "studio")
        // A custom PATH variable wins, as before.
        #expect(environment["PATH"] == "/override")

        let withoutOverride = ExoProcessController.makeEnvironment(
            base: ["PATH": "/usr/bin:/custom/bin"], runtimeURL: runtime, settingsEnvironment: [:])
        let path = withoutOverride["PATH"]?.split(separator: ":").map(String.init) ?? []
        #expect(path.first == runtime.path)
        #expect(path.contains("/custom/bin"))
        #expect(path.filter { $0 == "/usr/bin" }.count == 1)
    }

    @MainActor
    @Test func theControllerReadsAndConvertsStoredSettings() {
        let storage = ScratchDefaults()
        storage.defaults.set("studio", forKey: "EXOCustomNamespace")
        storage.defaults.set(true, forKey: FastSynchSetting.legacyEnabledKey)
        storage.defaults.set("/a", forKey: ModelFolder.legacyWritableKey)

        let controller = ExoProcessController(readinessProbe: { false }, defaults: storage.defaults)

        #expect(controller.customNamespace == "studio")
        #expect(controller.fastSynch == .automatic)
        #expect(controller.otherModelFolders.map(\.path) == ["/a"])
        #expect(controller.launchSettings.environment(buildTag: "t")["EXO_FAST_SYNCH"] == nil)
    }

    @MainActor
    @Test func changesAreSavedAsTheyAreMade() {
        let storage = ScratchDefaults()
        let controller = ExoProcessController(readinessProbe: { false }, defaults: storage.defaults)

        controller.fastSynch = .on
        controller.otherModelFolders = [ModelFolder(path: "/b", readOnly: true)]
        controller.offlineMode = true

        let reloaded = ExoProcessController(readinessProbe: { false }, defaults: storage.defaults)
        #expect(reloaded.fastSynch == .on)
        #expect(reloaded.otherModelFolders.map(\.path) == ["/b"])
        #expect(reloaded.otherModelFolders.first?.readOnly == true)
        #expect(reloaded.offlineMode)
    }

    @MainActor
    @Test func aStoppedExoNeverNeedsRestarting() async throws {
        let storage = ScratchDefaults()
        let controller = ExoProcessController(readinessProbe: { false }, defaults: storage.defaults)
        controller.modelsRunningOnThisMac = { true }

        controller.offlineMode = true
        try await Task.sleep(for: ExoProcessController.settingsApplyDelay + .milliseconds(300))

        #expect(!controller.needsRestart)
        #expect(controller.status == .stopped)
    }
}

// MARK: - Restarting to apply settings

struct SettingsRestartPolicyTests {
    private let old = ["EXO_ZENOH_NAMESPACE": "1.0.71"]
    private let new = ["EXO_ZENOH_NAMESPACE": "studio"]

    @Test func nothingToDoWhenExoAlreadyHasTheSettings() {
        for models in [true, false, nil] as [Bool?] {
            #expect(
                SettingsRestartPolicy.decide(
                    status: .running, runningEnvironment: old, settingsEnvironment: old,
                    modelsRunningOnThisMac: models) == .nothing)
        }
    }

    @Test func aStoppedOrFailedExoGetsTheSettingsWhenItStarts() {
        for status in [ExoProcessController.Status.stopped, .failed(message: "x")] {
            #expect(
                SettingsRestartPolicy.decide(
                    status: status, runningEnvironment: nil, settingsEnvironment: new,
                    modelsRunningOnThisMac: true) == .nothing)
        }
    }

    @Test func restartsAtOnceWhenNothingOnThisMacWouldStop() {
        #expect(
            SettingsRestartPolicy.decide(
                status: .running, runningEnvironment: old, settingsEnvironment: new,
                modelsRunningOnThisMac: false) == .restartNow)
        #expect(
            SettingsRestartPolicy.decide(
                status: .starting, runningEnvironment: old, settingsEnvironment: new,
                modelsRunningOnThisMac: nil) == .restartNow)
    }

    @Test func asksFirstWhenModelsRunOnThisMacOrThatIsntKnown() {
        for models in [true, nil] as [Bool?] {
            #expect(
                SettingsRestartPolicy.decide(
                    status: .running, runningEnvironment: old, settingsEnvironment: new,
                    modelsRunningOnThisMac: models) == .ask)
        }
    }

    @Test func modelsRunOnThisMacWhenAnInstanceHasARunnerHere() {
        let here = ClusterInstance(
            instanceId: "a",
            shardAssignments: ShardAssignments(
                modelId: "m", nodeToRunner: ["this-mac": "r1", "other": "r2"]),
            sharding: "Pipeline")
        let elsewhere = ClusterInstance(
            instanceId: "b",
            shardAssignments: ShardAssignments(modelId: "m", nodeToRunner: ["other": "r3"]),
            sharding: "Pipeline")

        #expect(ClusterStateService.modelsRun(on: "this-mac", in: [elsewhere, here]))
        #expect(!ClusterStateService.modelsRun(on: "this-mac", in: [elsewhere]))
        #expect(!ClusterStateService.modelsRun(on: "this-mac", in: [ClusterInstance]()))
    }
}

// MARK: - Settings copy and helpers

struct SettingsHelperTests {

    @Test func invalidVariableNamesAreNamedAndSaidToBeSkipped() {
        #expect(AdvancedSettingsPane.invalidNamesHint([]) == nil)
        let one = AdvancedSettingsPane.invalidNamesHint(["1BAD"]) ?? ""
        #expect(one.hasPrefix("“1BAD” isn't a valid name, so it isn't passed to EXO."))
        let two = AdvancedSettingsPane.invalidNamesHint(["A-B", "1C"]) ?? ""
        #expect(two.hasPrefix("“A-B”, “1C” aren't valid names, so they aren't passed to EXO."))
    }

    @Test func diagnosticsUseWordsAndSymbolsAndCopyAsText() {
        let network = NetworkStatus(
            thunderboltBridgeState: .enabled,
            bridgeInactive: nil,
            interfaceStatuses: [
                InterfaceIpStatus(interfaceName: "en0", ipAddress: nil),
                InterfaceIpStatus(interfaceName: "en1", ipAddress: "192.0.2.10"),
                InterfaceIpStatus(interfaceName: "en2", ipAddress: nil),
            ],
            localRdmaDevices: [],
            localRdmaActivePorts: [])

        let summary = DiagnosticsSummary.make(network: network, cluster: nil, localNodeId: nil)

        let bridge = summary.categories[0].items[0]
        #expect(bridge.value == "On")
        #expect(bridge.mark == .warning)
        #expect(
            summary.text == """
                This Mac:
                  Thunderbolt Bridge: On
                Network interfaces:
                  en1: 192.0.2.10
                  No address: en0, en2
                """)
    }
}
