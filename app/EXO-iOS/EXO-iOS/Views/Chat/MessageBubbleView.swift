import SwiftUI

struct MessageBubbleView: View {
    let message: ChatMessage

    private var isAssistant: Bool { message.role == .assistant }

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 48) }

            VStack(alignment: isAssistant ? .leading : .trailing, spacing: 6) {
                // Header
                HStack(spacing: 4) {
                    if isAssistant {
                        Circle()
                            .fill(Color.exoYellow)
                            .frame(width: 6, height: 6)
                            .shadow(color: Color.exoYellow.opacity(0.6), radius: 4)
                        Text("EXO")
                            .font(.exoMono(10, weight: .bold))
                            .tracking(1.5)
                            .foregroundStyle(Color.exoYellow)
                    } else {
                        Text("QUERY")
                            .font(.exoMono(10, weight: .medium))
                            .tracking(1.5)
                            .foregroundStyle(Color.exoLightGray)
                    }
                }

                // Bubble
                HStack(spacing: 0) {
                    if isAssistant {
                        RoundedRectangle(cornerRadius: 1)
                            .fill(Color.exoYellow.opacity(0.5))
                            .frame(width: 2)
                    }

                    Text(message.content + (message.isStreaming ? " \u{258C}" : ""))
                        .font(.exoBody)
                        .textSelection(.enabled)
                        .foregroundStyle(Color.exoForeground)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                }
                .background(Color.exoDarkGray)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            if isAssistant { Spacer(minLength: 48) }
        }
    }
}
