import SwiftUI

@main
struct EXO_iOSApp: App {
    @State private var clusterService = ClusterService()
    @State private var discoveryService = DiscoveryService()
    @State private var localInferenceService = LocalInferenceService()
    @State private var chatService: ChatService?

    var body: some Scene {
        WindowGroup {
            if let chatService {
                RootView()
                    .environment(clusterService)
                    .environment(discoveryService)
                    .environment(chatService)
                    .environment(localInferenceService)
                    .task {
                        await clusterService.attemptAutoReconnect()
                        discoveryService.startBrowsing()
                        await localInferenceService.prepareModel()
                    }
                    .onChange(of: discoveryService.discoveredClusters) { _, clusters in
                        guard !clusterService.isConnected,
                              case .disconnected = clusterService.connectionState,
                              let first = clusters.first
                        else { return }
                        Task {
                            await clusterService.connectToDiscoveredCluster(first, using: discoveryService)
                        }
                    }
            } else {
                Color.clear.onAppear {
                    chatService = ChatService(
                        clusterService: clusterService,
                        localInferenceService: localInferenceService
                    )
                }
            }
        }
    }
}
