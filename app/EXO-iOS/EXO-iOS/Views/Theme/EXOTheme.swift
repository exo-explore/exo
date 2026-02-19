import SwiftUI

// MARK: - EXO Color Palette

extension Color {
    /// Primary background — near-black (#121212)
    static let exoBlack = Color(red: 0x12 / 255.0, green: 0x12 / 255.0, blue: 0x12 / 255.0)
    /// Card / surface background (#1F1F1F)
    static let exoDarkGray = Color(red: 0x1F / 255.0, green: 0x1F / 255.0, blue: 0x1F / 255.0)
    /// Input field / elevated surface (#353535)
    static let exoMediumGray = Color(red: 0x35 / 255.0, green: 0x35 / 255.0, blue: 0x35 / 255.0)
    /// Secondary text (#999999)
    static let exoLightGray = Color(red: 0x99 / 255.0, green: 0x99 / 255.0, blue: 0x99 / 255.0)
    /// Accent yellow — matches dashboard (#FFD700)
    static let exoYellow = Color(red: 0xFF / 255.0, green: 0xD7 / 255.0, blue: 0x00 / 255.0)
    /// Primary foreground text (#E5E5E5)
    static let exoForeground = Color(red: 0xE5 / 255.0, green: 0xE5 / 255.0, blue: 0xE5 / 255.0)
    /// Destructive / error (#E74C3C)
    static let exoDestructive = Color(red: 0xE7 / 255.0, green: 0x4C / 255.0, blue: 0x3C / 255.0)
}

// MARK: - EXO Typography (SF Mono via .monospaced design)

extension Font {
    /// Monospaced font at a given size and weight.
    static func exoMono(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }

    /// Body text — 15pt monospaced
    static let exoBody: Font = .system(size: 15, weight: .regular, design: .monospaced)
    /// Caption — 11pt monospaced
    static let exoCaption: Font = .system(size: 11, weight: .regular, design: .monospaced)
    /// Subheadline — 13pt monospaced medium
    static let exoSubheadline: Font = .system(size: 13, weight: .medium, design: .monospaced)
    /// Headline — 17pt monospaced semibold
    static let exoHeadline: Font = .system(size: 17, weight: .semibold, design: .monospaced)
}

// MARK: - Reusable Gradient Divider

struct GradientDivider: View {
    var body: some View {
        LinearGradient(
            colors: [.clear, Color.exoYellow.opacity(0.3), .clear],
            startPoint: .leading,
            endPoint: .trailing
        )
        .frame(height: 1)
    }
}
