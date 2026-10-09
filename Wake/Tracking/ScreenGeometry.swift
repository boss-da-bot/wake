import AppKit
import CoreGraphics

/// Quartz window bounds are top-left, y down, origin at the top-left of the
/// primary display. Cocoa screen frames are bottom-left, y up.
enum ScreenGeometry {
    static func cocoaRect(fromQuartzWindowBounds quartz: CGRect) -> CGRect {
        let primaryTop = NSScreen.screens.first?.frame.maxY ?? quartz.height
        return CGRect(
            x: quartz.origin.x,
            y: primaryTop - quartz.origin.y - quartz.height,
            width: quartz.width,
            height: quartz.height
        )
    }

    static func displayID(of screen: NSScreen) -> CGDirectDisplayID? {
        let key = NSDeviceDescriptionKey("NSScreenNumber")
        guard let number = screen.deviceDescription[key] as? NSNumber else {
            return nil
        }
        return CGDirectDisplayID(number.uint32Value)
    }

    static func screen(for displayID: CGDirectDisplayID, in screens: [NSScreen]) -> NSScreen? {
        screens.first { self.displayID(of: $0) == displayID }
    }

    /// A window covers the desktop when it fills the visible frame (fullscreen
    /// or zoomed). The menu bar and Dock sit outside `visibleFrame`.
    static func coversDesktop(_ frame: CGRect, screen: NSScreen) -> Bool {
        let visible = screen.visibleFrame
        let slop: CGFloat = 12
        return frame.minX <= visible.minX + slop
            && frame.minY <= visible.minY + slop
            && frame.maxX >= visible.maxX - slop
            && frame.maxY >= visible.maxY - slop
    }
}
