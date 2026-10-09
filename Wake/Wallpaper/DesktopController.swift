import AppKit
import simd

/// One wallpaper window per screen. Windows are recreated when displays change.
final class DesktopController {
    private let gpu: FluidGPU
    private var windows: [CGDirectDisplayID: DesktopWindow] = [:]

    init(gpu: FluidGPU) {
        self.gpu = gpu
    }

    func sync(with screens: [NSScreen], palette: Palette) {
        var seen = Set<CGDirectDisplayID>()
        for screen in screens {
            guard let displayID = ScreenGeometry.displayID(of: screen) else { continue }
            seen.insert(displayID)
            if let existing = windows[displayID] {
                existing.updateFrame(to: screen)
                existing.applyBackground(palette.background)
            } else {
                let window = DesktopWindow(screen: screen, gpu: gpu, displayID: displayID)
                window.applyBackground(palette.background)
                windows[displayID] = window
            }
        }
        for displayID in windows.keys where !seen.contains(displayID) {
            windows[displayID]?.close()
            windows.removeValue(forKey: displayID)
        }
    }

    func render(
        sample: MotionSample,
        dt: Float,
        iterations: Int,
        drift: Float,
        palette: Palette
    ) {
        for (displayID, window) in windows {
            if sample.coveredDisplays.contains(displayID) { continue }
            window.render(
                splats: sample.splatsByDisplay[displayID] ?? [],
                obstacles: sample.obstaclesByDisplay[displayID] ?? [],
                dt: dt,
                iterations: iterations,
                drift: drift,
                background: palette.background
            )
        }
    }

    func apply(palette: Palette) {
        for window in windows.values {
            window.applyBackground(palette.background)
        }
    }
}

private extension DesktopWindow {
    func render(
        splats: [Splat],
        obstacles: [Obstacle],
        dt: Float,
        iterations: Int,
        drift: Float,
        background: SIMD3<Float>
    ) {
        fluidView?.render(
            splats: splats,
            obstacles: obstacles,
            dt: dt,
            iterations: iterations,
            drift: drift,
            background: background
        )
    }
}
