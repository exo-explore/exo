import Foundation

/// The settings that become exo's environment. exo reads them only when it
/// starts, so a change takes effect after exo restarts.
struct ExoLaunchSettings: Equatable {
    /// Empty means automatic: exo's own cluster name, the same for every
    /// version. exo keeps devices of another build apart itself, and names
    /// them, instead of them silently forming another cluster.
    var clusterName = ""
    var huggingFaceToken = ""
    var huggingFaceEndpoint = ""
    var showImageModels = false
    var offline = false
    var fastSynch = FastSynchSetting.automatic
    /// Empty means exo's default, ~/.exo/models.
    var downloadFolder = ""
    var otherModelFolders: [ModelFolder] = []
    var customEnvironmentVariables: [CustomEnvironmentVariable] = []

    /// The variables these settings give exo, applied over the app's own
    /// environment, with `buildID`: exo connects only to devices running the
    /// same build. The custom variables come last so they can override the
    /// others. Invalid custom variable names are left out.
    func environment(buildID: String) -> [String: String] {
        var environment: [String: String] = ["EXO_BUILD_ID": buildID]
        let name = clusterName.trimmingCharacters(in: .whitespaces)
        if !name.isEmpty {
            environment["EXO_ZENOH_NAMESPACE"] = name
        }
        if !huggingFaceToken.isEmpty {
            environment["HF_TOKEN"] = huggingFaceToken
        }
        // Also fixes up a value saved before Settings normalised it.
        let endpoint = HuggingFaceEndpoint.normalized(huggingFaceEndpoint)
        if !endpoint.isEmpty {
            environment["HF_ENDPOINT"] = endpoint
        }
        if showImageModels {
            environment["EXO_ENABLE_IMAGE_MODELS"] = "true"
        }
        if offline {
            environment["EXO_OFFLINE"] = "true"
        }
        if let fastSynch = fastSynch.environmentValue {
            environment["EXO_FAST_SYNCH"] = fastSynch
        }
        let download = downloadFolder.trimmingCharacters(in: .whitespaces)
        if !download.isEmpty {
            environment["EXO_DEFAULT_MODELS_DIR"] = download
        }
        environment.merge(ModelFolder.environment(for: otherModelFolders)) { _, folders in
            folders
        }
        for variable in CustomEnvironmentVariable.sanitized(customEnvironmentVariables) {
            environment[variable.key] = variable.value
        }
        return environment
    }
}
