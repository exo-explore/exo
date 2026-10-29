import AppKit

/// The menu bar icon: EXO's pixel "exo" wordmark as a template image (black
/// and clear, which the system colours for the menu bar), with a badge when
/// something needs attention and a dot when an update is ready.
enum MenuBarGlyph {
    /// The wordmark's pixels, row by row from the top, taken from the logo
    /// artwork (a 20 x 5 grid).
    static let wordmark = [
        ".####..##..##..####.",
        "##..##..####..##..##",
        "######...##...##..##",
        "##......####..##..##",
        ".#####.##..##..####.",
    ]

    /// Points per wordmark pixel. Whole device pixels at 1x and 2x keep the
    /// edges crisp; the wordmark is 40 x 10 pt, as tall as menu bar text.
    static let pixelSize: CGFloat = 2

    /// Room for the badges at the right edge, which overlap the "o".
    static let size = NSSize(width: 44, height: 20)

    static var wordmarkRect: NSRect {
        let width = CGFloat(wordmark[0].count) * pixelSize
        let height = CGFloat(wordmark.count) * pixelSize
        return NSRect(x: 0, y: ((size.height - height) / 2).rounded(), width: width, height: height)
    }

    /// Bottom right, over the corner of the "o".
    static let attentionBadgeRect = NSRect(x: 34, y: 0, width: 10, height: 10)
    /// The exclamation mark cut out of the badge, on whole points so it
    /// stays sharp at 1x: a bar and a dot.
    static let exclamationBar = NSRect(x: 38, y: 4, width: 2, height: 4)
    static let exclamationDot = NSRect(x: 38, y: 1, width: 2, height: 2)
    /// Top right.
    static let updateDotRect = NSRect(x: 39, y: 14, width: 5, height: 5)
    /// The clear ring cut around a badge so it reads as separate.
    static let badgeGap: CGFloat = 1

    /// The icon for `style`, with the wordmark drawn at `alpha` (badges are
    /// always opaque so they stay visible on a dimmed icon).
    static func image(style: MenuBarIconStyle, alpha: Double) -> NSImage {
        let image = NSImage(size: size, flipped: false) { _ in
            NSColor.black.withAlphaComponent(alpha).setFill()
            wordmarkPath().fill()

            if style.needsAttention {
                knockOut(
                    NSBezierPath(ovalIn: attentionBadgeRect.insetBy(dx: -badgeGap, dy: -badgeGap)))
                NSColor.black.setFill()
                NSBezierPath(ovalIn: attentionBadgeRect).fill()
                knockOut(NSBezierPath(rect: exclamationBar))
                knockOut(NSBezierPath(rect: exclamationDot))
            }
            if style.updateAvailable {
                knockOut(NSBezierPath(ovalIn: updateDotRect.insetBy(dx: -badgeGap, dy: -badgeGap)))
                NSColor.black.setFill()
                NSBezierPath(ovalIn: updateDotRect).fill()
            }
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = style.accessibilityLabel
        return image
    }

    /// One path for all the wordmark's pixels, so neighbouring squares
    /// don't show seams.
    static func wordmarkPath() -> NSBezierPath {
        let path = NSBezierPath()
        let origin = wordmarkRect.origin
        for (row, line) in wordmark.enumerated() {
            for (column, pixel) in line.enumerated() where pixel == "#" {
                path.appendRect(
                    NSRect(
                        x: origin.x + CGFloat(column) * pixelSize,
                        y: origin.y + CGFloat(wordmark.count - 1 - row) * pixelSize,
                        width: pixelSize, height: pixelSize))
            }
        }
        return path
    }

    /// Clears `path` out of what has been drawn so far.
    private static func knockOut(_ path: NSBezierPath) {
        guard let graphicsContext = NSGraphicsContext.current else { return }
        graphicsContext.saveGraphicsState()
        graphicsContext.compositingOperation = .destinationOut
        NSColor.black.setFill()
        path.fill()
        graphicsContext.restoreGraphicsState()
    }
}
