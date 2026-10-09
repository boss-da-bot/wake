import AppKit
import simd

/// Borderless, click-through window parked just under the desktop icons.
final class DesktopWindow: NSWindow {
    let displayID: CGDirectDisplayID
    let fluidView: FluidView

    init?(screen: NSScreen, gpu: FluidGPU, displayID: CGDirectDisplayID) {
        self.displayID = displayID
        let view = FluidView(gpu: gpu, screen: screen)
        fluidView = view
        super.init(
            contentRect: screen.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false,
            screen: screen
        )
        isReleasedWhenClosed = false
        isOpaque = true
        hasShadow = false
        ignoresMouseEvents = true
        acceptsMouseMovedEvents = false
        isMovable = false
        isMovableByWindowBackground = false
        isExcludedFromWindowsMenu = true
        isRestorable = false
        title = "Wake"
        canHide = false
        hidesOnDeactivate = false
        animationBehavior = .none
        appearance = NSAppearance(named: .darkAqua)
        backgroundColor = .black
        collectionBehavior = Self.spaceBehavior
        level = Self.wallpaperLevel
        contentView = view
        view.frame = contentView?.bounds ?? screen.frame
        view.autoresizingMask = [.width, .height]
        // Stay invisible until the level and click-through flags are in place.
        alphaValue = 0
        orderFrontRegardless()
        level = Self.wallpaperLevel
        ignoresMouseEvents = true
        collectionBehavior = Self.spaceBehavior
        orderBack(nil)
        alphaValue = 1
    }

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    /// One level under the desktop icons, which is above the system wallpaper.
    static var wallpaperLevel: NSWindow.Level {
        let icons = CGWindowLevelForKey(.desktopIconWindow)
        return NSWindow.Level(rawValue: Int(icons) - 1)
    }

    static let spaceBehavior: NSWindow.CollectionBehavior = [
        .canJoinAllSpaces,
        .stationary,
        .ignoresCycle
    ]

    func updateFrame(to screen: NSScreen) {
        setFrame(screen.frame, display: true)
        fluidView.updateGrid(for: screen.frame.size)
        level = Self.wallpaperLevel
        ignoresMouseEvents = true
        collectionBehavior = Self.spaceBehavior
    }

    func applyBackground(_ color: SIMD3<Float>) {
        backgroundColor = NSColor(
            calibratedRed: CGFloat(color.x),
            green: CGFloat(color.y),
            blue: CGFloat(color.z),
            alpha: 1
        )
        fluidView.applyBackground(color)
    }
}
