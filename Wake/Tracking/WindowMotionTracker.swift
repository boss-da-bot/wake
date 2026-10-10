import AppKit
import CoreGraphics
import simd

struct MotionSample {
    var splatsByDisplay: [CGDirectDisplayID: [Splat]]
    var coveredDisplays: Set<CGDirectDisplayID>
    var interacting: Bool
    var obstaclesByDisplay: [CGDirectDisplayID: [Obstacle]]
}

/// Diffs on-screen window bounds and turns the motion into fluid splats.
///
/// `CGWindowListCopyWindowInfo` is used for frames and window ids only.
/// Wake never asks for Screen Recording and never copies window pixels.
final class WindowMotionTracker {
    private var previous: [CGWindowID: CGRect] = [:]
    private var seeded = false
    private var lastCursor: CGPoint?
    private var lastTime: CFTimeInterval?
    private(set) var lastInteractionTime: CFTimeInterval = 0

    private enum Tuning {
        static let minimumEdgeSpeed: Float = 90
        static let edgeSpacing: CGFloat = 84
        static let outsidePadding: CGFloat = 28
        static let maxWindowSplats = 63
        static let massChangeCount = 4
        static let forceGain: Float = 0.40
        static let minimumWindowSize: CGFloat = 48
        static let maxObstacles = 48
        static let maxObstacleSpeed: Float = 4
    }

    func rebase(now: CFTimeInterval) {
        let windows = Self.copyWindows()
        previous = Self.framesByID(windows)
        seeded = true
        lastTime = now
        lastCursor = NSEvent.mouseLocation
    }

    func sample(now: CFTimeInterval, screens: [NSScreen], palette: Palette, intensity: Float) -> MotionSample {
        let windows = Self.copyWindows()
        let cursor = NSEvent.mouseLocation
        let covered = coveredDisplays(windows: windows, screens: screens)
        let dt: Float
        if let lastTime {
            dt = Float(max(now - lastTime, 1.0 / 240.0))
        } else {
            dt = 1.0 / 60.0
        }

        defer {
            previous = Self.framesByID(windows)
            lastCursor = cursor
            lastTime = now
            seeded = true
        }

        let currentIDs = Set(windows.map(\.id))
        let appeared = windows.filter { previous[$0.id] == nil }
        let disappeared = previous.filter { !currentIDs.contains($0.key) }
        let massChange = seeded
            && (appeared.count >= Tuning.massChangeCount || disappeared.count >= Tuning.massChangeCount)
        // A Space switch teleports windows. Keep them solid, but don't treat the
        // jump as a body moving through the pool.
        let obstacles = obstaclesByDisplay(
            windows: windows,
            screens: screens,
            covered: covered,
            dt: dt,
            allowMotion: seeded && !massChange
        )

        guard seeded else {
            return MotionSample(
                splatsByDisplay: [:],
                coveredDisplays: covered,
                interacting: false,
                obstaclesByDisplay: obstacles
            )
        }

        var splatsByDisplay: [CGDirectDisplayID: [Splat]] = [:]
        var interacting = false

        if !massChange {
            for screen in screens {
                guard let displayID = ScreenGeometry.displayID(of: screen) else { continue }
                if covered.contains(displayID) { continue }
                var splats: [Splat] = []

                for window in windows where window.frame.intersects(screen.frame) {
                    if ScreenGeometry.coversDesktop(window.frame, screen: screen) { continue }
                    let color = palette.color(forWindow: window.id)
                    if let old = previous[window.id] {
                        let stirred = edgeSplats(
                            previous: old,
                            current: window.frame,
                            dt: dt,
                            screen: screen,
                            color: color,
                            intensity: intensity
                        )
                        if !stirred.isEmpty {
                            interacting = true
                            splats.append(contentsOf: stirred)
                        }
                    } else {
                        let puff = puffSplats(frame: window.frame, screen: screen, color: color, intensity: intensity)
                        if !puff.isEmpty {
                            interacting = true
                            splats.append(contentsOf: puff)
                        }
                    }
                }

                for (id, frame) in disappeared where frame.intersects(screen.frame) {
                    if ScreenGeometry.coversDesktop(frame, screen: screen) { continue }
                    let color = palette.color(forWindow: id)
                    let puff = puffSplats(frame: frame, screen: screen, color: color, intensity: intensity * 0.85)
                    if !puff.isEmpty {
                        interacting = true
                        splats.append(contentsOf: puff)
                    }
                }

                splatsByDisplay[displayID] = downsample(splats, limit: Tuning.maxWindowSplats)
            }
        }

        if let lastCursor, let cursorSplat = cursorSplat(
            from: lastCursor,
            to: cursor,
            dt: dt,
            screens: screens,
            covered: covered,
            palette: palette,
            intensity: intensity
        ) {
            interacting = true
            splatsByDisplay[cursorSplat.display, default: []].append(cursorSplat.splat)
        }

        if interacting {
            lastInteractionTime = now
        }

        return MotionSample(
            splatsByDisplay: splatsByDisplay,
            coveredDisplays: covered,
            interacting: interacting,
            obstaclesByDisplay: obstacles
        )
    }

    /// Every tracked window is a solid. Motion is the edge velocity so a resize
    /// pushes on the edges that actually moved, not only the center.
    private func obstaclesByDisplay(
        windows: [ListedWindow],
        screens: [NSScreen],
        covered: Set<CGDirectDisplayID>,
        dt: Float,
        allowMotion: Bool
    ) -> [CGDirectDisplayID: [Obstacle]] {
        var result: [CGDirectDisplayID: [Obstacle]] = [:]
        for screen in screens {
            guard let displayID = ScreenGeometry.displayID(of: screen) else { continue }
            if covered.contains(displayID) { continue }
            var ranked: [(area: CGFloat, obstacle: Obstacle)] = []
            ranked.reserveCapacity(windows.count)
            for window in windows where window.frame.intersects(screen.frame) {
                if ScreenGeometry.coversDesktop(window.frame, screen: screen) { continue }
                guard let obstacle = makeObstacle(
                    previous: previous[window.id],
                    current: window.frame,
                    screen: screen,
                    dt: dt,
                    allowMotion: allowMotion
                ) else { continue }
                ranked.append((window.frame.width * window.frame.height, obstacle))
            }
            if ranked.count > Tuning.maxObstacles {
                ranked.sort { $0.area > $1.area }
                ranked.removeLast(ranked.count - Tuning.maxObstacles)
            }
            result[displayID] = ranked.map(\.obstacle)
        }
        return result
    }

    private func makeObstacle(
        previous: CGRect?,
        current: CGRect,
        screen: NSScreen,
        dt: Float,
        allowMotion: Bool
    ) -> Obstacle? {
        let frame = screen.frame
        guard frame.width > 1, frame.height > 1 else { return nil }
        let minX = Float((current.minX - frame.minX) / frame.width)
        let minY = Float((current.minY - frame.minY) / frame.height)
        let maxX = Float((current.maxX - frame.minX) / frame.width)
        let maxY = Float((current.maxY - frame.minY) / frame.height)
        guard maxX > minX, maxY > minY else { return nil }

        var motion = SIMD4<Float>(0, 0, 0, 0)
        if allowMotion, let previous, dt > 0 {
            let invW = 1 / Float(frame.width)
            let invH = 1 / Float(frame.height)
            motion = SIMD4(
                clampObstacle((Float(current.minX - previous.minX) * invW) / dt),
                clampObstacle((Float(current.minY - previous.minY) * invH) / dt),
                clampObstacle((Float(current.width - previous.width) * invW) / dt),
                clampObstacle((Float(current.height - previous.height) * invH) / dt)
            )
        }
        return Obstacle(rect: SIMD4(minX, minY, maxX, maxY), motion: motion)
    }

    private func clampObstacle(_ value: Float) -> Float {
        min(Tuning.maxObstacleSpeed, max(-Tuning.maxObstacleSpeed, value))
    }

    private func coveredDisplays(windows: [ListedWindow], screens: [NSScreen]) -> Set<CGDirectDisplayID> {
        var covered: Set<CGDirectDisplayID> = []
        for screen in screens {
            guard let displayID = ScreenGeometry.displayID(of: screen) else { continue }
            let covers = windows.contains { ScreenGeometry.coversDesktop($0.frame, screen: screen) }
            if covers {
                covered.insert(displayID)
            }
        }
        return covered
    }

    private func edgeSplats(
        previous: CGRect,
        current: CGRect,
        dt: Float,
        screen: NSScreen,
        color: SIMD3<Float>,
        intensity: Float
    ) -> [Splat] {
        let frame = screen.frame
        guard frame.width > 1, frame.height > 1, dt > 0, intensity > 0 else { return [] }

        let dOrigin = SIMD2(
            Float(current.origin.x - previous.origin.x),
            Float(current.origin.y - previous.origin.y)
        )
        let dSize = SIMD2(
            Float(current.width - previous.width),
            Float(current.height - previous.height)
        )
        let travel = hypot(dOrigin.x, dOrigin.y)
        let grew = hypot(dSize.x, dSize.y)
        if travel < Float(1) && grew < Float(1) { return [] }

        let leftV = SIMD2(dOrigin.x, dOrigin.y + dSize.y * 0.5) / dt
        let rightV = SIMD2(dOrigin.x + dSize.x, dOrigin.y + dSize.y * 0.5) / dt
        let bottomV = SIMD2(dOrigin.x + dSize.x * 0.5, dOrigin.y) / dt
        let topV = SIMD2(dOrigin.x + dSize.x * 0.5, dOrigin.y + dSize.y) / dt

        let edges: [(CGPoint, CGPoint, SIMD2<Float>, SIMD2<Float>)] = [
            (CGPoint(x: current.minX, y: current.minY), CGPoint(x: current.maxX, y: current.minY), SIMD2(0, -1), bottomV),
            (CGPoint(x: current.maxX, y: current.minY), CGPoint(x: current.maxX, y: current.maxY), SIMD2(1, 0), rightV),
            (CGPoint(x: current.maxX, y: current.maxY), CGPoint(x: current.minX, y: current.maxY), SIMD2(0, 1), topV),
            (CGPoint(x: current.minX, y: current.maxY), CGPoint(x: current.minX, y: current.minY), SIMD2(-1, 0), leftV)
        ]

        var splats: [Splat] = []
        let screenWidth = Float(frame.width)
        let screenHeight = Float(frame.height)

        for edge in edges {
            let edgeLength = hypot(edge.1.x - edge.0.x, edge.1.y - edge.0.y)
            let speed = hypot(edge.3.x, edge.3.y)
            if edgeLength < CGFloat(1) || speed < Tuning.minimumEdgeSpeed { continue }
            let count = max(1, Int((edgeLength / Tuning.edgeSpacing).rounded()))
            let speedUV = hypot(edge.3.x / screenWidth, edge.3.y / screenHeight)
            let direction = edge.3 / speed
            let facing = direction.x * edge.2.x + direction.y * edge.2.y
            let lead = max(0, facing)
            let trail = max(0, -facing)
            let side = 1 - abs(facing)
            let response = pow(min(speedUV, 3.5 as Float), 0.72 as Float)
            let velocityDirection = SIMD2(edge.3.x / screenWidth, edge.3.y / screenHeight) / max(speedUV, 1e-4)
            let normalUV = normalize(SIMD2(edge.2.x / screenWidth, edge.2.y / screenHeight))
            // Motion only displaces ink that is already there.
            var impulse = velocityDirection * response * Tuning.forceGain * intensity * (0.45 * lead + 0.20 * trail)
            impulse += normalUV * response * 0.12 * intensity * side
            let radius = 0.022 + min(0.02, response * 0.012)
            let offset = SIMD2(
                edge.2.x * Float(Tuning.outsidePadding / frame.width),
                edge.2.y * Float(Tuning.outsidePadding / frame.height)
            )

            for index in 0..<count {
                let t = (CGFloat(index) + 0.5) / CGFloat(count)
                let x = edge.0.x + (edge.1.x - edge.0.x) * t
                let y = edge.0.y + (edge.1.y - edge.0.y) * t
                let position = SIMD2(
                    Float((x - frame.minX) / frame.width),
                    Float((y - frame.minY) / frame.height)
                ) + offset
                guard position.x > -0.08, position.x < 1.08, position.y > -0.08, position.y < 1.08 else { continue }
                splats.append(Splat(position: position, radius: radius, impulse: impulse, color: color, amount: 0))
            }
        }

        return splats
    }

    private func puffSplats(frame: CGRect, screen: NSScreen, color: SIMD3<Float>, intensity: Float) -> [Splat] {
        let screenFrame = screen.frame
        guard screenFrame.width > 1, screenFrame.height > 1, intensity > 0 else { return [] }
        let area = frame.width * frame.height
        let screenArea = screenFrame.width * screenFrame.height
        if area > screenArea * 0.82 { return [] }

        let center = SIMD2(
            Float((frame.midX - screenFrame.minX) / screenFrame.width),
            Float((frame.midY - screenFrame.minY) / screenFrame.height)
        )
        guard center.x > -0.05, center.x < 1.05, center.y > -0.05, center.y < 1.05 else { return [] }

        let half = SIMD2(
            Float(frame.width / screenFrame.width) * 0.5,
            Float(frame.height / screenFrame.height) * 0.5
        )
        let count = 8
        var splats: [Splat] = []
        splats.reserveCapacity(count)
        for index in 0..<count {
            let angle = Float(index) / Float(count) * Float.pi * 2
            let direction = SIMD2(cos(angle), sin(angle))
            let position = center + SIMD2(direction.x * (half.x + 0.02), direction.y * (half.y + 0.02))
            let impulse = direction * 0.20 * intensity
            splats.append(Splat(
                position: position,
                radius: 0.046,
                impulse: impulse,
                color: color,
                amount: 0
            ))
        }
        return splats
    }

    private func cursorSplat(
        from start: CGPoint,
        to end: CGPoint,
        dt: Float,
        screens: [NSScreen],
        covered: Set<CGDirectDisplayID>,
        palette: Palette,
        intensity: Float
    ) -> (display: CGDirectDisplayID, splat: Splat)? {
        let delta = SIMD2(Float(end.x - start.x), Float(end.y - start.y))
        if length(delta) < Float(2.5) || intensity <= 0 || dt <= 0 { return nil }
        guard let screen = screens.first(where: { $0.frame.contains(end) }),
              let displayID = ScreenGeometry.displayID(of: screen),
              !covered.contains(displayID) else { return nil }
        let frame = screen.frame
        let speed = hypot(delta.x / Float(frame.width), delta.y / Float(frame.height)) / dt
        if speed < Float(0.08) { return nil }
        let position = SIMD2(
            Float((end.x - frame.minX) / frame.width),
            Float((end.y - frame.minY) / frame.height)
        )
        let impulse = SIMD2(delta.x / Float(frame.width), delta.y / Float(frame.height)) / dt * 0.07 * intensity
        let color = palette.inks[0]
        let splat = Splat(
            position: position,
            radius: 0.018,
            impulse: impulse,
            color: color,
            amount: 0
        )
        return (displayID, splat)
    }

    private func downsample(_ splats: [Splat], limit: Int) -> [Splat] {
        guard splats.count > limit, limit > 0 else { return splats }
        var kept: [Splat] = []
        kept.reserveCapacity(limit)
        let stride = Float(splats.count) / Float(limit)
        for index in 0..<limit {
            let source = min(splats.count - 1, Int(Float(index) * stride))
            kept.append(splats[source])
        }
        return kept
    }

    private struct ListedWindow {
        var id: CGWindowID
        var frame: CGRect
    }

    /// Bounds only. Titles, thumbnails, and screen captures are intentionally unused.
    private static func copyWindows() -> [ListedWindow] {
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let raw = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else {
            return []
        }
        let myPID = Int(ProcessInfo.processInfo.processIdentifier)
        var windows: [ListedWindow] = []
        windows.reserveCapacity(raw.count)
        for info in raw {
            if intValue(info[kCGWindowOwnerPID as String]) == myPID { continue }
            let layer = intValue(info[kCGWindowLayer as String]) ?? 0
            // Normal windows and floating panels. Skip the menu bar, Dock, and desktop.
            if layer < 0 || layer > 8 { continue }
            let alpha = doubleValue(info[kCGWindowAlpha as String]) ?? 1
            if alpha < 0.05 { continue }
            guard let id = intValue(info[kCGWindowNumber as String]), id > 0,
                  let quartz = cgRect(info[kCGWindowBounds as String]) else { continue }
            if quartz.width < Tuning.minimumWindowSize || quartz.height < Tuning.minimumWindowSize { continue }
            guard let windowID = UInt32(exactly: id) else { continue }
            let cocoa = ScreenGeometry.cocoaRect(fromQuartzWindowBounds: quartz)
            windows.append(ListedWindow(id: windowID, frame: cocoa))
        }
        return windows
    }

    private static func framesByID(_ windows: [ListedWindow]) -> [CGWindowID: CGRect] {
        var frames: [CGWindowID: CGRect] = [:]
        frames.reserveCapacity(windows.count)
        for window in windows {
            frames[window.id] = window.frame
        }
        return frames
    }

    private static func cgRect(_ value: Any?) -> CGRect? {
        // NSDictionary is the bridged form of the bounds dictionary. A conditional
        // cast to CFDictionary is rejected by the compiler because it always succeeds.
        guard let dictionary = value as? NSDictionary else { return nil }
        var rect = CGRect.null
        guard CGRectMakeWithDictionaryRepresentation(dictionary, &rect), !rect.isNull, !rect.isInfinite else {
            return nil
        }
        return rect
    }

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let number as Int:
            return number
        case let number as Int32:
            return Int(number)
        case let number as UInt32:
            return Int(number)
        case let number as UInt:
            return Int(number)
        case let number as NSNumber:
            return number.intValue
        default:
            return nil
        }
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let number as Double:
            return number
        case let number as Float:
            return Double(number)
        case let number as NSNumber:
            return number.doubleValue
        default:
            return nil
        }
    }
}
