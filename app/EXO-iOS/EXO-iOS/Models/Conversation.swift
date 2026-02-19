import Foundation

struct Conversation: Identifiable, Codable, Equatable {
    let id: UUID
    var title: String
    var messages: [StoredMessage]
    var modelId: String?
    let createdAt: Date

    init(
        id: UUID = UUID(), title: String = "New Chat", messages: [StoredMessage] = [],
        modelId: String? = nil, createdAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.modelId = modelId
        self.createdAt = createdAt
    }
}

struct StoredMessage: Identifiable, Codable, Equatable {
    let id: UUID
    let role: String
    var content: String
    let timestamp: Date

    init(id: UUID = UUID(), role: String, content: String, timestamp: Date = Date()) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
    }
}
