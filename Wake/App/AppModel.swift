import AppKit
import QuartzCore

/// Owns settings, the frame clock, and the per-screen wallpapers.
final class AppModel: NSObject {
    let settings = SettingsStore()
    let power = PowerMonitor()
    let tracker = WindowMotionTracker()
    private let clock = FrameClock()
    private let menu = MenuBarController()
    private var desktop: DesktopController?
    private var currentFPS = 0
    private var wasFullyCovered = false
    private var displayAsleep = false
    private var lastTick: CFTimeInterval = 0

    private(set) var palette: Palette
    private(set) var intensity: Double
    private(set) var isPaused: Bool
    let gpu: FluidGPU?

    var metalAvailable: Bool { gpu != nil }

    override init() {
        // Read through a local store so this stays legal before super.init.
        // `settings` uses the same UserDefaults and is what later writes go through.
        let stored = SettingsStore()
        palette = Palette.named(stored.paletteID)
        intensity = stored.intensity
        isPaused = stored.isPaused
        gpu = FluidGPU()
        super.init()
    }

    func start() {
        power.refreshIfNeeded(force: true)
        if let gpu {
            let desktop = DesktopController(gpu: gpu)
            desktop.sync(with: NSScreen.screens, palette: palette)
            self.desktop = desktop
        } else {
            Log.app.error("Wake is running without a wallpaper because Metal is unavailable")
        }

        menu.install(model: self)
        lastTick = CACurrentMediaTime()
        observe()
        clock.start { [weak self] in
            self?.tick()
        }
        setFPS(isPaused || gpu == nil ? 1 : 30)
    }

    func togglePaused() {
        isPaused.toggle()
        settings.isPaused = isPaused
        menu.applyPaused(isPaused)
        guard !isPaused else {
            setFPS(1)
            return
        }
        lastTick = CACurrentMediaTime()
        tracker.rebase(now: lastTick)
        setFPS(60)
    }

    func setPalette(_ palette: Palette) {
        self.palette = palette
        settings.paletteID = palette.id
        desktop?.apply(palette: palette)
    }

    func setIntensity(_ value: Double) {
        intensity = min(1, max(0, value))
        settings.intensity = intensity
    }

    private func observe() {
        let center = NotificationCenter.default
        center.addObserver(
            self,
            selector: #selector(screensChanged),
            name: NSApplication.didChangeScreenParametersNotification,
            object: nil
        )
        center.addObserver(
            self,
            selector: #selector(powerChanged),
            name: Notification.Name.NSProcessInfoPowerStateDidChange,
            object: nil
        )
        let workspace = NSWorkspace.shared.notificationCenter
        workspace.addObserver(
            self,
            selector: #selector(screensDidSleep),
            name: NSWorkspace.screensDidSleepNotification,
            object: nil
        )
        workspace.addObserver(
            self,
            selector: #selector(screensDidWake),
            name: NSWorkspace.screensDidWakeNotification,
            object: nil
        )
    }

    private func tick() {
        guard gpu != nil else {
            setFPS(1)
            return
        }
        power.refreshIfNeeded()
        let now = CACurrentMediaTime()
        let dt = Float(min(max(now - lastTick, 0), 0.05))
        lastTick = now

        if displayAsleep || isPaused {
            setFPS(1)
            return
        }

        let screens = NSScreen.screens
        let sample = tracker.sample(
            now: now,
            screens: screens,
            palette: palette,
            intensity: Float(intensity)
        )
        let fullyCovered = Self.coversAll(screens, sample: sample)
        if fullyCovered {
            wasFullyCovered = true
            setFPS(2)
            return
        }

        let leavingCoverage = wasFullyCovered
        wasFullyCovered = false
        if leavingCoverage {
            tracker.rebase(now: now)
        }

        let fps = targetFPS(now: now)
        setFPS(fps)
        var presented = sample
        if leavingCoverage {
            presented.splatsByDisplay = [:]
            presented.interacting = false
        }
        desktop?.render(
            sample: presented,
            dt: dt,
            iterations: iterations(for: fps),
            drift: driftStrength(now: now),
            palette: palette,
            intensity: Float(intensity)
        )
    }

    private static func coversAll(_ screens: [NSScreen], sample: MotionSample) -> Bool {
        guard !screens.isEmpty else { return false }
        return screens.allSatisfy { screen in
            guard let id = ScreenGeometry.displayID(of: screen) else { return false }
            return sample.coveredDisplays.contains(id)
        }
    }

    private func targetFPS(now: CFTimeInterval) -> Int {
        let since = now - tracker.lastInteractionTime
        let cap: Int
        if power.isThermalLimited || power.isLowPowerMode {
            cap = 15
        } else if power.isOnBattery {
            cap = 30
        } else {
            cap = 60
        }
        let desired: Int
        if since < 1.5 {
            desired = 60
        } else if since < 8 {
            desired = 24
        } else {
            desired = 10
        }
        return min(desired, cap)
    }

    private func iterations(for fps: Int) -> Int {
        if power.prefersReducedRate { return 8 }
        if fps <= 10 { return 6 }
        return 16
    }

    private func driftStrength(now: CFTimeInterval) -> Float {
        let since = now - tracker.lastInteractionTime
        if since > 8 { return 0.07 }
        if since > 1.5 { return 0.03 }
        return 0.012
    }

    private func setFPS(_ fps: Int) {
        guard fps != currentFPS else { return }
        currentFPS = fps
        clock.setFramesPerSecond(fps)
    }

    @objc private func screensChanged() {
        desktop?.sync(with: NSScreen.screens, palette: palette)
        tracker.rebase(now: CACurrentMediaTime())
    }

    @objc private func powerChanged() {
        power.refreshIfNeeded(force: true)
    }

    @objc private func screensDidSleep() {
        displayAsleep = true
        setFPS(1)
    }

    @objc private func screensDidWake() {
        displayAsleep = false
        lastTick = CACurrentMediaTime()
        desktop?.sync(with: NSScreen.screens, palette: palette)
        tracker.rebase(now: lastTick)
        setFPS(30)
    }
}

private final class FrameClock {
    private var timer: DispatchSourceTimer?
    private var fps = 0

    func start(handler: @escaping () -> Void) {
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.setEventHandler(handler: handler)
        self.timer = timer
        setFramesPerSecond(30)
        timer.resume()
    }

    func setFramesPerSecond(_ newValue: Int) {
        let clamped = min(60, max(1, newValue))
        guard clamped != fps, let timer else { return }
        fps = clamped
        let interval = 1.0 / Double(clamped)
        let leeway = max(1, Int((interval * 1000) / 5))
        timer.schedule(deadline: .now(), repeating: interval, leeway: .milliseconds(leeway))
    }
}
