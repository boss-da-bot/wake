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
    private var heldSample: MotionSample?
    private var lastPoll: CFTimeInterval = 0
    private var statsFrames = 0
    private var statsGPU = 0.0
    private var statsGPUCount = 0
    private var statsStart: CFTimeInterval = 0
    private(set) var showsFrameStats = false
    private(set) var frameStatsLine = ""

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
        heldSample = nil
        tracker.rebase(now: lastTick)
        setFPS(60)
    }

    func toggleFrameStats() {
        showsFrameStats.toggle()
        if !showsFrameStats {
            frameStatsLine = ""
        }
        menu.refresh()
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
        let dt = Float(min(max(now - lastTick, 0), 0.25))
        lastTick = now

        if displayAsleep || isPaused {
            setFPS(1)
            return
        }

        let screens = NSScreen.screens
        let sample = motionSample(now: now, screens: screens)
        let fullyCovered = Self.coversAll(screens, sample: sample)
        if fullyCovered {
            wasFullyCovered = true
            setFPS(1)
            return
        }

        let leavingCoverage = wasFullyCovered
        wasFullyCovered = false
        if leavingCoverage {
            tracker.rebase(now: now)
            lastPoll = now
        }

        let pace = pace(at: now)
        let fps = targetFPS(pace)
        setFPS(fps)
        var presented = sample
        if leavingCoverage {
            presented.splatsByDisplay = [:]
            presented.interacting = false
        }
        let gpuMilliseconds = desktop?.render(
            sample: presented,
            dt: dt,
            iterations: iterations(for: pace),
            drift: driftStrength(pace),
            palette: palette,
            intensity: Float(intensity)
        ) ?? 0
        noteFrame(gpuMilliseconds: gpuMilliseconds, now: now)
    }

    /// Window-list polls are the idle CPU cost, so they are not tied to the
    /// render rate. Between polls the last solids stay put and splats are
    /// not applied again.
    private func motionSample(now: CFTimeInterval, screens: [NSScreen]) -> MotionSample {
        let interval = pollInterval(pace(at: now))
        if heldSample == nil || now - lastPoll >= interval {
            let fresh = tracker.sample(
                now: now,
                screens: screens,
                palette: palette,
                intensity: Float(intensity)
            )
            heldSample = fresh
            lastPoll = now
            return fresh
        }
        var held = heldSample ?? MotionSample(
            splatsByDisplay: [:],
            coveredDisplays: [],
            interacting: false,
            obstaclesByDisplay: [:]
        )
        held.splatsByDisplay = [:]
        held.interacting = false
        return held
    }

    private static func coversAll(_ screens: [NSScreen], sample: MotionSample) -> Bool {
        guard !screens.isEmpty else { return false }
        return screens.allSatisfy { screen in
            guard let id = ScreenGeometry.displayID(of: screen) else { return false }
            return sample.coveredDisplays.contains(id)
        }
    }

    private enum Pace {
        case active
        case settling
        case idle
    }

    private func pace(at now: CFTimeInterval) -> Pace {
        let since = now - tracker.lastInteractionTime
        if since < 1.2 { return .active }
        if since < 5 { return .settling }
        return .idle
    }

    private func pollInterval(_ pace: Pace) -> CFTimeInterval {
        switch pace {
        case .active: return 0
        case .settling: return 0.25
        case .idle: return 0.5
        }
    }

    private func targetFPS(_ pace: Pace) -> Int {
        let reduced = power.isThermalLimited || power.isLowPowerMode
        switch pace {
        case .active:
            if reduced { return 20 }
            if power.isOnBattery { return 30 }
            return 60
        case .settling:
            if reduced { return 8 }
            if power.isOnBattery { return 12 }
            return 18
        case .idle:
            if reduced { return 4 }
            if power.isOnBattery { return 6 }
            return 8
        }
    }

    private func iterations(for pace: Pace) -> Int {
        let reduced = power.isThermalLimited || power.isLowPowerMode
        switch pace {
        case .active:
            return reduced ? 10 : 16
        case .settling:
            return reduced ? 6 : 8
        case .idle:
            return 4
        }
    }

    private func driftStrength(_ pace: Pace) -> Float {
        switch pace {
        case .active: return 0.012
        case .settling: return 0.03
        case .idle: return 0.07
        }
    }

    private func noteFrame(gpuMilliseconds: Double, now: CFTimeInterval) {
        if statsStart == 0 { statsStart = now }
        statsFrames += 1
        if gpuMilliseconds > 0 {
            statsGPU += gpuMilliseconds
            statsGPUCount += 1
        }
        let elapsed = now - statsStart
        guard elapsed >= 2 else { return }
        let fps = Double(statsFrames) / elapsed
        let gpu = statsGPUCount > 0 ? statsGPU / Double(statsGPUCount) : 0
        if gpu > 0 {
            frameStatsLine = String(format: "%.0f fps · %.1f ms GPU", fps, gpu)
        } else {
            frameStatsLine = String(format: "%.0f fps", fps)
        }
        if showsFrameStats {
            Log.app.info("\(self.frameStatsLine, privacy: .public)")
            menu.refresh()
        }
        statsFrames = 0
        statsGPU = 0
        statsGPUCount = 0
        statsStart = now
    }

    private func setFPS(_ fps: Int) {
        guard fps != currentFPS else { return }
        currentFPS = fps
        clock.setFramesPerSecond(fps)
    }

    @objc private func screensChanged() {
        desktop?.sync(with: NSScreen.screens, palette: palette)
        heldSample = nil
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
        heldSample = nil
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
