import SwiftUI

/// EXO's Settings window, shown by the app's `Settings` scene: a toolbar of
/// panes, titled after the pane shown, reopening on the pane last shown.
/// Every change is saved as it is made; settings that exo reads when it
/// starts are applied by restarting it (see `ExoProcessController`).
struct SettingsView: View {
    @AppStorage(SettingsPane.storageKey) private var pane = SettingsPane.general

    var body: some View {
        TabView(selection: $pane) {
            Tab("General", systemImage: "gearshape", value: SettingsPane.general) {
                GeneralSettingsPane()
            }
            Tab("Cluster", systemImage: "network", value: SettingsPane.cluster) {
                ClusterSettingsPane()
            }
            Tab("Models", systemImage: "cube", value: SettingsPane.models) {
                ModelsSettingsPane()
            }
            Tab("Advanced", systemImage: "gearshape.2", value: SettingsPane.advanced) {
                AdvancedSettingsPane()
            }
        }
    }
}
