import SwiftUI

struct RootView: View {
    @Environment(ClusterService.self) private var clusterService
    @Environment(DiscoveryService.self) private var discoveryService
    @Environment(ChatService.self) private var chatService
    @Environment(LocalInferenceService.self) private var localInferenceService
    @State private var showSettings = false
    @State private var showConversations = false

    var body: some View {
        NavigationStack {
            ChatView()
                .navigationTitle("EXO")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        conversationMenuButton
                    }

                    ToolbarItem(placement: .principal) {
                        ConnectionStatusBadge(
                            connectionState: clusterService.connectionState,
                            localModelState: localInferenceService.modelState
                        )
                    }

                    ToolbarItem(placement: .topBarTrailing) {
                        Button {
                            showSettings = true
                        } label: {
                            Image(systemName: "gear")
                        }
                    }
                }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView()
                .environment(discoveryService)
        }
        .sheet(isPresented: $showConversations) {
            conversationList
        }
    }

    // MARK: - Conversations

    private var conversationMenuButton: some View {
        HStack(spacing: 12) {
            Button {
                showConversations = true
            } label: {
                Image(systemName: "sidebar.left")
            }

            Button {
                chatService.createConversation()
            } label: {
                Image(systemName: "square.and.pencil")
            }
        }
    }

    private var conversationList: some View {
        NavigationStack {
            List {
                if chatService.conversations.isEmpty {
                    Text("No conversations yet")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(chatService.conversations) { conversation in
                        Button {
                            chatService.setActiveConversation(id: conversation.id)
                            showConversations = false
                        } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(conversation.title)
                                    .fontWeight(
                                        conversation.id == chatService.activeConversationId
                                            ? .semibold : .regular
                                    )
                                    .lineLimit(1)

                                if let modelId = conversation.modelId {
                                    Text(modelId)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                            }
                        }
                        .tint(.primary)
                    }
                    .onDelete { indexSet in
                        for index in indexSet {
                            chatService.deleteConversation(id: chatService.conversations[index].id)
                        }
                    }
                }
            }
            .navigationTitle("Conversations")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { showConversations = false }
                }
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        chatService.createConversation()
                    } label: {
                        Image(systemName: "plus")
                    }
                }
            }
        }
    }
}
