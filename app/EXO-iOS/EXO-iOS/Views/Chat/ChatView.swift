import SwiftUI

struct ChatView: View {
    @Environment(ClusterService.self) private var clusterService
    @Environment(ChatService.self) private var chatService
    @Environment(LocalInferenceService.self) private var localInferenceService
    @State private var inputText = ""
    @State private var showModelSelector = false

    var body: some View {
        VStack(spacing: 0) {
            modelBar

            Divider()

            messageList

            Divider()

            inputBar
        }
        .sheet(isPresented: $showModelSelector) {
            ModelSelectorView(
                models: clusterService.availableModels,
                selectedModelId: chatService.activeConversation?.modelId
            ) { modelId in
                chatService.setModelForActiveConversation(modelId)
            }
        }
    }

    // MARK: - Model Bar

    private var useLocalModel: Bool {
        !clusterService.isConnected && localInferenceService.isAvailable
    }

    private var modelBar: some View {
        Button {
            if !useLocalModel {
                showModelSelector = true
            }
        } label: {
            HStack {
                Image(systemName: useLocalModel ? "iphone" : "cpu")
                    .foregroundStyle(useLocalModel ? .blue : .secondary)

                if useLocalModel {
                    Text(localInferenceService.defaultModelId)
                        .font(.subheadline)
                        .lineLimit(1)
                } else if let modelId = chatService.activeConversation?.modelId {
                    Text(modelId)
                        .font(.subheadline)
                        .lineLimit(1)
                } else {
                    Text("Select Model")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if useLocalModel {
                    Text("On-Device")
                        .font(.caption2)
                        .foregroundStyle(.blue)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.blue.opacity(0.1))
                        .clipShape(Capsule())
                } else {
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 10)
            .background(Color(.secondarySystemBackground))
        }
        .tint(.primary)
        .disabled(useLocalModel)
    }

    // MARK: - Messages

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    if chatService.activeMessages.isEmpty {
                        emptyState
                    } else {
                        ForEach(chatService.activeMessages) { message in
                            MessageBubbleView(message: message)
                                .id(message.id)
                        }
                    }
                }
                .padding()
            }
            .onChange(of: chatService.activeMessages.last?.content) {
                if let lastId = chatService.activeMessages.last?.id {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(lastId, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer(minLength: 80)
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text("Start a conversation")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Send a message to begin chatting with the model.")
                .font(.subheadline)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
            Spacer(minLength: 80)
        }
        .padding()
    }

    // MARK: - Input

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message...", text: $inputText, axis: .vertical)
                .lineLimit(1...6)
                .textFieldStyle(.plain)
                .padding(10)
                .background(Color(.tertiarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 20))

            if chatService.isGenerating {
                Button {
                    chatService.cancelGeneration()
                } label: {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.red)
                }
            } else {
                Button {
                    let text = inputText
                    inputText = ""
                    chatService.sendMessage(text)
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(canSend ? Color.accentColor : Color.gray)
                }
                .disabled(!canSend)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && (clusterService.isConnected || localInferenceService.isAvailable)
    }
}
