import Foundation

// MARK: - Request

struct ChatCompletionRequest: Encodable {
    let model: String
    let messages: [ChatCompletionMessageParam]
    let stream: Bool
    let maxTokens: Int?
    let temperature: Double?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature
        case maxTokens = "max_tokens"
    }
}

struct ChatCompletionMessageParam: Encodable {
    let role: String
    let content: String
}

// MARK: - Streaming Response

struct ChatCompletionChunk: Decodable {
    let id: String
    let model: String?
    let choices: [StreamingChoice]
    let usage: ChunkUsage?

    init(id: String, model: String?, choices: [StreamingChoice], usage: ChunkUsage?) {
        self.id = id
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

struct StreamingChoice: Decodable {
    let index: Int
    let delta: Delta
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }

    init(index: Int, delta: Delta, finishReason: String?) {
        self.index = index
        self.delta = delta
        self.finishReason = finishReason
    }
}

struct Delta: Decodable {
    let role: String?
    let content: String?

    init(role: String?, content: String?) {
        self.role = role
        self.content = content
    }
}

struct ChunkUsage: Decodable {
    let promptTokens: Int?
    let completionTokens: Int?
    let totalTokens: Int?

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }

    init(promptTokens: Int?, completionTokens: Int?, totalTokens: Int?) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
    }
}

// MARK: - Non-Streaming Response

struct ChatCompletionResponse: Decodable {
    let id: String
    let model: String?
    let choices: [ResponseChoice]
}

struct ResponseChoice: Decodable {
    let index: Int
    let message: ResponseMessage
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index, message
        case finishReason = "finish_reason"
    }
}

struct ResponseMessage: Decodable {
    let role: String?
    let content: String?
}

// MARK: - Models List

struct ModelListResponse: Decodable {
    let data: [ModelInfo]
}

struct ModelInfo: Decodable, Identifiable {
    let id: String
    let name: String?
}

// MARK: - Error

struct APIErrorResponse: Decodable {
    let error: APIErrorInfo
}

struct APIErrorInfo: Decodable {
    let message: String
    let type: String?
    let code: Int?
}
