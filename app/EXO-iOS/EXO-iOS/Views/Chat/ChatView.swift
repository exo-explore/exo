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

            GradientDivider()

            messageList

            GradientDivider()

            inputBar
        }
        .background(Color.exoBlack)
        .sheet(isPresented: $showModelSelector) {
            ModelSelectorView(
                models: clusterService.availableModels,
                selectedModelId: chatService.activeConversation?.modelId
            ) { modelId in
                chatService.setModelForActiveConversation(modelId)
            }
            .presentationBackground(Color.exoDarkGray)
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
                    .font(.exoCaption)
                    .foregroundStyle(useLocalModel ? Color.exoYellow : Color.exoLightGray)

                if useLocalModel {
                    Text(localInferenceService.defaultModelId)
                        .font(.exoSubheadline)
                        .foregroundStyle(Color.exoForeground)
                        .lineLimit(1)
                } else if let modelId = chatService.activeConversation?.modelId {
                    Text(modelId)
                        .font(.exoSubheadline)
                        .foregroundStyle(Color.exoForeground)
                        .lineLimit(1)
                } else {
                    Text("SELECT MODEL")
                        .font(.exoSubheadline)
                        .tracking(1.5)
                        .foregroundStyle(Color.exoLightGray)
                }

                Spacer()

                if useLocalModel {
                    Text("ON-DEVICE")
                        .font(.exoCaption)
                        .tracking(1)
                        .foregroundStyle(Color.exoYellow)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.exoYellow.opacity(0.15))
                        .clipShape(Capsule())
                } else {
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundStyle(Color.exoLightGray)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 10)
            .background(Color.exoDarkGray)
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
            .background(Color.exoBlack)
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
        VStack(spacing: 16) {
            Spacer(minLength: 80)

            ZStack {
                Circle()
                    .stroke(Color.exoYellow.opacity(0.15), lineWidth: 1)
                    .frame(width: 80, height: 80)
                Circle()
                    .stroke(Color.exoYellow.opacity(0.3), lineWidth: 1)
                    .frame(width: 56, height: 56)
                Circle()
                    .fill(Color.exoYellow.opacity(0.15))
                    .frame(width: 32, height: 32)
                Circle()
                    .fill(Color.exoYellow)
                    .frame(width: 8, height: 8)
                    .shadow(color: Color.exoYellow.opacity(0.6), radius: 6)
            }

            Text("AWAITING INPUT")
                .font(.exoSubheadline)
                .tracking(3)
                .foregroundStyle(Color.exoLightGray)

            Text("Send a message to begin.")
                .font(.exoCaption)
                .foregroundStyle(Color.exoLightGray.opacity(0.6))

            Spacer(minLength: 80)
        }
        .padding()
    }

    // MARK: - Input

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message...", text: $inputText, axis: .vertical)
                .font(.exoBody)
                .lineLimit(1...6)
                .textFieldStyle(.plain)
                .padding(10)
                .background(Color.exoMediumGray)
                .foregroundStyle(Color.exoForeground)
                .clipShape(RoundedRectangle(cornerRadius: 8))

            if chatService.isGenerating {
                Button {
                    chatService.cancelGeneration()
                } label: {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundStyle(Color.exoDestructive)
                }
            } else {
                Button {
                    let text = inputText
                    inputText = ""
                    chatService.sendMessage(text)
                } label: {
                    Text("SEND")
                        .font(.exoMono(12, weight: .bold))
                        .tracking(1)
                        .foregroundStyle(canSend ? Color.exoBlack : Color.exoLightGray)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(canSend ? Color.exoYellow : Color.exoMediumGray)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .disabled(!canSend)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color.exoDarkGray)
    }

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && (clusterService.isConnected || localInferenceService.isAvailable)
    }
}
