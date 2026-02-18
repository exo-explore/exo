import Foundation

@Observable
@MainActor
final class ChatService {
    var conversations: [Conversation] = []
    var activeConversationId: UUID?
    private(set) var isGenerating: Bool = false
    private var currentGenerationTask: Task<Void, Never>?

    private let clusterService: ClusterService
    private let localInferenceService: LocalInferenceService

    var canSendMessage: Bool {
        clusterService.isConnected || localInferenceService.isAvailable
    }

    var activeConversation: Conversation? {
        guard let id = activeConversationId else { return nil }
        return conversations.first { $0.id == id }
    }

    var activeMessages: [ChatMessage] {
        guard let conversation = activeConversation else { return [] }
        return conversation.messages.map { stored in
            ChatMessage(
                id: stored.id,
                role: ChatMessage.Role(rawValue: stored.role) ?? .user,
                content: stored.content,
                timestamp: stored.timestamp
            )
        }
    }

    init(clusterService: ClusterService, localInferenceService: LocalInferenceService) {
        self.clusterService = clusterService
        self.localInferenceService = localInferenceService
        loadConversations()
    }

    // MARK: - Conversation Management

    func createConversation(modelId: String? = nil) {
        let conversation = Conversation(
            modelId: modelId ?? clusterService.availableModels.first?.id)
        conversations.insert(conversation, at: 0)
        activeConversationId = conversation.id
        saveConversations()
    }

    func deleteConversation(id: UUID) {
        conversations.removeAll { $0.id == id }
        if activeConversationId == id {
            activeConversationId = conversations.first?.id
        }
        saveConversations()
    }

    func setActiveConversation(id: UUID) {
        activeConversationId = id
    }

    func setModelForActiveConversation(_ modelId: String) {
        guard let index = conversations.firstIndex(where: { $0.id == activeConversationId }) else {
            return
        }
        conversations[index].modelId = modelId
        saveConversations()
    }

    // MARK: - Messaging

    func sendMessage(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        if activeConversation == nil {
            createConversation()
        }

        guard let index = conversations.firstIndex(where: { $0.id == activeConversationId }) else {
            return
        }

        let userMessage = StoredMessage(role: "user", content: text)
        conversations[index].messages.append(userMessage)

        if conversations[index].title == "New Chat" {
            let preview = String(text.prefix(40))
            conversations[index].title = preview + (text.count > 40 ? "..." : "")
        }

        let modelId: String
        if clusterService.isConnected {
            guard let clusterId = conversations[index].modelId ?? clusterService.availableModels.first?.id
            else {
                let errorMessage = StoredMessage(
                    role: "assistant", content: "No model selected. Please select a model first.")
                conversations[index].messages.append(errorMessage)
                saveConversations()
                return
            }
            modelId = clusterId
        } else if localInferenceService.isAvailable {
            modelId = localInferenceService.defaultModelId
        } else {
            let errorMessage = StoredMessage(
                role: "assistant",
                content: "Not connected to a cluster and local model is not available.")
            conversations[index].messages.append(errorMessage)
            saveConversations()
            return
        }

        conversations[index].modelId = modelId

        let assistantMessageId = UUID()
        let assistantMessage = StoredMessage(
            id: assistantMessageId, role: "assistant", content: "", timestamp: Date())
        conversations[index].messages.append(assistantMessage)

        let messagesForAPI = conversations[index].messages.dropLast().map { stored in
            ChatCompletionMessageParam(role: stored.role, content: stored.content)
        }

        let request = ChatCompletionRequest(
            model: modelId,
            messages: Array(messagesForAPI),
            stream: true,
            maxTokens: 4096,
            temperature: nil
        )

        let conversationId = conversations[index].id

        isGenerating = true
        currentGenerationTask = Task { [weak self] in
            guard let self else { return }
            await self.performStreaming(
                request: request, conversationId: conversationId,
                assistantMessageId: assistantMessageId)
        }

        saveConversations()
    }

    func cancelGeneration() {
        currentGenerationTask?.cancel()
        currentGenerationTask = nil
        localInferenceService.cancelGeneration()
        isGenerating = false
    }

    // MARK: - Streaming

    private func performStreaming(
        request: ChatCompletionRequest, conversationId: UUID, assistantMessageId: UUID
    ) async {
        defer {
            isGenerating = false
            currentGenerationTask = nil
            saveConversations()
        }

        do {
            let stream =
                clusterService.isConnected
                ? clusterService.streamChatCompletion(request: request)
                : localInferenceService.streamChatCompletion(request: request)
            for try await chunk in stream {
                guard !Task.isCancelled else { return }
                guard let content = chunk.choices.first?.delta.content, !content.isEmpty else {
                    continue
                }

                if let convIndex = conversations.firstIndex(where: { $0.id == conversationId }),
                    let msgIndex = conversations[convIndex].messages.firstIndex(where: {
                        $0.id == assistantMessageId
                    })
                {
                    conversations[convIndex].messages[msgIndex].content += content
                }
            }
        } catch {
            if !Task.isCancelled {
                if let convIndex = conversations.firstIndex(where: { $0.id == conversationId }),
                    let msgIndex = conversations[convIndex].messages.firstIndex(where: {
                        $0.id == assistantMessageId
                    })
                {
                    if conversations[convIndex].messages[msgIndex].content.isEmpty {
                        conversations[convIndex].messages[msgIndex].content =
                            "Error: \(error.localizedDescription)"
                    }
                }
            }
        }
    }

    // MARK: - Persistence

    private static var storageURL: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            .first!
        return documents.appendingPathComponent("exo_conversations.json")
    }

    private func saveConversations() {
        do {
            let data = try JSONEncoder().encode(conversations)
            try data.write(to: Self.storageURL, options: .atomic)
        } catch {
            // Save failed silently
        }
    }

    private func loadConversations() {
        do {
            let data = try Data(contentsOf: Self.storageURL)
            conversations = try JSONDecoder().decode([Conversation].self, from: data)
            activeConversationId = conversations.first?.id
        } catch {
            conversations = []
        }
    }
}
