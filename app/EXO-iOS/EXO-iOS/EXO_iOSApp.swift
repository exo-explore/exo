import SwiftUI
import UIKit

@main
struct EXO_iOSApp: App {
    @State private var clusterService = ClusterService()
    @State private var discoveryService = DiscoveryService()
    @State private var localInferenceService = LocalInferenceService()
    @State private var chatService: ChatService?

    init() {
        let darkGray = UIColor(red: 0x1F / 255.0, green: 0x1F / 255.0, blue: 0x1F / 255.0, alpha: 1)
        let yellow = UIColor(red: 0xFF / 255.0, green: 0xD7 / 255.0, blue: 0x00 / 255.0, alpha: 1)

        let navAppearance = UINavigationBarAppearance()
        navAppearance.configureWithOpaqueBackground()
        navAppearance.backgroundColor = darkGray
        navAppearance.titleTextAttributes = [
            .foregroundColor: yellow,
            .font: UIFont.monospacedSystemFont(ofSize: 17, weight: .semibold),
        ]
        navAppearance.largeTitleTextAttributes = [
            .foregroundColor: yellow,
            .font: UIFont.monospacedSystemFont(ofSize: 34, weight: .bold),
        ]

        UINavigationBar.appearance().standardAppearance = navAppearance
        UINavigationBar.appearance().compactAppearance = navAppearance
        UINavigationBar.appearance().scrollEdgeAppearance = navAppearance
        UINavigationBar.appearance().tintColor = yellow
    }

    var body: some Scene {
        WindowGroup {
            if let chatService {
                RootView()
                    .environment(clusterService)
                    .environment(discoveryService)
                    .environment(chatService)
                    .environment(localInferenceService)
                    .preferredColorScheme(.dark)
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
                            await clusterService.connectToDiscoveredCluster(
                                first, using: discoveryService)
                        }
                    }
            } else {
                Color.exoBlack.onAppear {
                    chatService = ChatService(
                        clusterService: clusterService,
                        localInferenceService: localInferenceService
                    )
                }
            }
        }
    }
}
