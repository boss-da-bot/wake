import AppKit
import simd

/// Borderless, click-through window parked just under the desktop icons.
final class DesktopWindow: NSWindow {
    let displayID: CGDirectDisplayID
    let fluidView: FluidView?

    private struct Seed {
        let displayID: CGDirectDisplayID
        let view: FluidView
        let screen: NSScreen
    }

    /// Arguments for the designated initializer. AppKit can re-enter it without
    /// our extra parameters, so they travel beside `self` instead of through it.
    private static var seeds: [Seed] = []

    convenience init(screen: NSScreen, gpu: FluidGPU, displayID: CGDirectDisplayID) {
        let view = FluidView(gpu: gpu, screen: screen)
        Self.seeds.append(Seed(displayID: displayID, view: view, screen: screen))
        defer { _ = Self.seeds.popLast() }
        // Do not call `init(...screen:)`. Its implementation messages
        // `init(contentRect:styleMask:backing:defer:)` on self, and Swift's
        // synthesized version of that designated initializer traps
        // (EXC_BREAKPOINT) when the subclass has stored properties.
        self.init(
            contentRect: screen.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
    }

    override init(
        contentRect: NSRect,
        styleMask style: NSWindow.StyleMask,
        backing backingStoreType: NSWindow.BackingStoreType,
        defer flag: Bool
    ) {
        let seed = Self.seeds.last
        displayID = seed?.displayID ?? 0
        fluidView = seed?.view
        super.init(
            contentRect: contentRect,
            styleMask: style,
            backing: backingStoreType,
            defer: flag
        )
        if let seed {
            installWallpaper(view: seed.view, screen: seed.screen)
        }
    }

    /// Nibs are not used. Failing the decode avoids a second synthesized trap.
    required init?(coder: NSCoder) {
        return nil
    }

    private func installWallpaper(view: FluidView, screen: NSScreen) {
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
        setFrame(screen.frame, display: false)
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
        fluidView?.updateGrid(for: screen.frame.size)
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
        fluidView?.applyBackground(color)
    }
}
