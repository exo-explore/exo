import Foundation
import MLXLLM
import MLXLMCommon

enum LocalModelState: Equatable {
    case notDownloaded
    case downloading(progress: Double)
    case downloaded
    case loading
    case ready
    case generating
    case error(String)
}

@Observable
@MainActor
final class LocalInferenceService {
    private(set) var modelState: LocalModelState = .notDownloaded
    private var modelContainer: ModelContainer?
    private var generationTask: Task<Void, Never>?

    let defaultModelId = "mlx-community/Qwen3-0.6B-4bit"

    private static let modelDownloadedKey = "exo_local_model_downloaded"

    var isReady: Bool {
        modelState == .ready
    }

    var isAvailable: Bool {
        modelState == .ready || modelState == .generating
    }

    init() {
        if UserDefaults.standard.bool(forKey: Self.modelDownloadedKey) {
            modelState = .downloaded
        }
    }

    // MARK: - Model Lifecycle

    func prepareModel() async {
        guard modelState == .notDownloaded || modelState == .downloaded else { return }

        let wasDownloaded = modelState == .downloaded

        if !wasDownloaded {
            modelState = .downloading(progress: 0)
        } else {
            modelState = .loading
        }

        do {
            let container = try await loadModelContainer(
                id: defaultModelId
            ) { [weak self] progress in
                guard let self else { return }
                Task { @MainActor in
                    if case .downloading = self.modelState {
                        self.modelState = .downloading(progress: progress.fractionCompleted)
                    }
                }
            }

            self.modelContainer = container
            UserDefaults.standard.set(true, forKey: Self.modelDownloadedKey)
            modelState = .ready
        } catch {
            modelState = .error(error.localizedDescription)
        }
    }

    func unloadModel() {
        cancelGeneration()
        modelContainer = nil
        modelState = .downloaded
    }

    // MARK: - Generation

    func streamChatCompletion(request: ChatCompletionRequest) -> AsyncThrowingStream<
        ChatCompletionChunk, Error
    > {
        AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish(throwing: LocalInferenceError.serviceUnavailable)
                    return
                }

                guard let container = self.modelContainer else {
                    continuation.finish(throwing: LocalInferenceError.modelNotLoaded)
                    return
                }

                await MainActor.run {
                    self.modelState = .generating
                }

                defer {
                    Task { @MainActor [weak self] in
                        if self?.modelState == .generating {
                            self?.modelState = .ready
                        }
                    }
                }

                let chunkId = "local-\(UUID().uuidString)"

                do {
                    // Build Chat.Message array from the request
                    var chatMessages: [Chat.Message] = []
                    for msg in request.messages {
                        switch msg.role {
                        case "system":
                            chatMessages.append(.system(msg.content))
                        case "assistant":
                            chatMessages.append(.assistant(msg.content))
                        default:
                            chatMessages.append(.user(msg.content))
                        }
                    }

                    // Use ChatSession for streaming generation
                    let session = ChatSession(
                        container,
                        history: chatMessages,
                        generateParameters: GenerateParameters(
                            maxTokens: request.maxTokens ?? 4096,
                            temperature: Float(request.temperature ?? 0.7)
                        )
                    )

                    // Stream with an empty prompt since history already contains the conversation
                    let stream = session.streamResponse(to: "")
                    for try await text in stream {
                        if Task.isCancelled { break }

                        let chunk = ChatCompletionChunk(
                            id: chunkId,
                            model: request.model,
                            choices: [
                                StreamingChoice(
                                    index: 0,
                                    delta: Delta(role: nil, content: text),
                                    finishReason: nil
                                )
                            ],
                            usage: nil
                        )
                        continuation.yield(chunk)
                    }

                    // Send final chunk with finish reason
                    let finalChunk = ChatCompletionChunk(
                        id: chunkId,
                        model: request.model,
                        choices: [
                            StreamingChoice(
                                index: 0,
                                delta: Delta(role: nil, content: nil),
                                finishReason: "stop"
                            )
                        ],
                        usage: nil
                    )
                    continuation.yield(finalChunk)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            self.generationTask = task

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        if modelState == .generating {
            modelState = .ready
        }
    }
}

enum LocalInferenceError: LocalizedError {
    case serviceUnavailable
    case modelNotLoaded

    var errorDescription: String? {
        switch self {
        case .serviceUnavailable: "Local inference service is unavailable"
        case .modelNotLoaded: "Local model is not loaded"
        }
    }
}
