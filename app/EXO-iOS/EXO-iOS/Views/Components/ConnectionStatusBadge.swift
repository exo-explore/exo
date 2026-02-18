import SwiftUI

struct ConnectionStatusBadge: View {
    let connectionState: ConnectionState
    var localModelState: LocalModelState = .notDownloaded

    private var isLocalReady: Bool {
        if case .disconnected = connectionState {
            return localModelState == .ready || localModelState == .generating
        }
        return false
    }

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(dotColor)
                .frame(width: 8, height: 8)

            Text(label)
                .font(.caption)
                .fontWeight(.medium)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(backgroundColor)
        .clipShape(Capsule())
    }

    private var dotColor: Color {
        if isLocalReady {
            return .blue
        }
        switch connectionState {
        case .connected: return .green
        case .connecting: return .orange
        case .disconnected: return .gray
        }
    }

    private var label: String {
        if isLocalReady {
            return "Local"
        }
        switch connectionState {
        case .connected: return "Connected"
        case .connecting: return "Connecting..."
        case .disconnected: return "Disconnected"
        }
    }

    private var backgroundColor: Color {
        if isLocalReady {
            return .blue.opacity(0.15)
        }
        switch connectionState {
        case .connected: return .green.opacity(0.15)
        case .connecting: return .orange.opacity(0.15)
        case .disconnected: return .gray.opacity(0.15)
        }
    }
}
