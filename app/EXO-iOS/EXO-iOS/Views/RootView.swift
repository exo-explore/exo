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
                                .foregroundStyle(Color.exoYellow)
                        }
                    }
                }
        }
        .tint(Color.exoYellow)
        .sheet(isPresented: $showSettings) {
            SettingsView()
                .environment(discoveryService)
                .presentationBackground(Color.exoDarkGray)
        }
        .sheet(isPresented: $showConversations) {
            conversationList
                .presentationBackground(Color.exoDarkGray)
        }
    }

    // MARK: - Conversations

    private var conversationMenuButton: some View {
        HStack(spacing: 12) {
            Button {
                showConversations = true
            } label: {
                Image(systemName: "sidebar.left")
                    .foregroundStyle(Color.exoYellow)
            }

            Button {
                chatService.createConversation()
            } label: {
                Image(systemName: "square.and.pencil")
                    .foregroundStyle(Color.exoYellow)
            }
        }
    }

    private var conversationList: some View {
        NavigationStack {
            List {
                if chatService.conversations.isEmpty {
                    Text("No conversations yet")
                        .font(.exoBody)
                        .foregroundStyle(Color.exoLightGray)
                        .listRowBackground(Color.exoDarkGray)
                } else {
                    ForEach(chatService.conversations) { conversation in
                        let isActive = conversation.id == chatService.activeConversationId
                        Button {
                            chatService.setActiveConversation(id: conversation.id)
                            showConversations = false
                        } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(conversation.title)
                                    .font(.exoSubheadline)
                                    .fontWeight(isActive ? .semibold : .regular)
                                    .foregroundStyle(
                                        isActive ? Color.exoYellow : Color.exoForeground
                                    )
                                    .lineLimit(1)

                                if let modelId = conversation.modelId {
                                    Text(modelId)
                                        .font(.exoCaption)
                                        .foregroundStyle(Color.exoLightGray)
                                        .lineLimit(1)
                                }
                            }
                        }
                        .listRowBackground(
                            isActive
                                ? Color.exoYellow.opacity(0.1)
                                : Color.exoDarkGray
                        )
                    }
                    .onDelete { indexSet in
                        for index in indexSet {
                            chatService.deleteConversation(id: chatService.conversations[index].id)
                        }
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .background(Color.exoBlack)
            .navigationTitle("Conversations")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { showConversations = false }
                        .font(.exoSubheadline)
                        .foregroundStyle(Color.exoYellow)
                }
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        chatService.createConversation()
                    } label: {
                        Image(systemName: "plus")
                            .foregroundStyle(Color.exoYellow)
                    }
                }
            }
        }
    }
}
