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
                .shadow(color: dotColor.opacity(0.6), radius: 4)

            Text(label.uppercased())
                .font(.exoMono(10, weight: .medium))
                .tracking(1)
                .foregroundStyle(Color.exoForeground)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(backgroundColor)
        .clipShape(Capsule())
        .overlay(
            Capsule()
                .stroke(dotColor.opacity(0.3), lineWidth: 1)
        )
    }

    private var dotColor: Color {
        if isLocalReady {
            return .exoYellow
        }
        switch connectionState {
        case .connected: return .green
        case .connecting: return .orange
        case .disconnected: return .exoLightGray
        }
    }

    private var label: String {
        if isLocalReady {
            return "Local"
        }
        switch connectionState {
        case .connected: return "Connected"
        case .connecting: return "Connecting"
        case .disconnected: return "Disconnected"
        }
    }

    private var backgroundColor: Color {
        if isLocalReady {
            return Color.exoYellow.opacity(0.1)
        }
        switch connectionState {
        case .connected: return .green.opacity(0.1)
        case .connecting: return .orange.opacity(0.1)
        case .disconnected: return Color.exoMediumGray.opacity(0.5)
        }
    }
}
