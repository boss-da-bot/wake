import AppKit
import MetalKit
import simd

final class FluidView: MTKView, MTKViewDelegate {
    private let gpu: FluidGPU?
    private var field: FluidField?
    private var pending: Pending?
    private var lastPresentedGPUMilliseconds: Double = 0

    private struct Pending {
        var splats: [Splat]
        var obstacles: [Obstacle]
        var dt: Float
        var iterations: Int
        var drift: Float
        var background: SIMD3<Float>
        var inks: [SIMD3<Float>]
        var intensity: Float
    }

    private struct Seed {
        let gpu: FluidGPU
        let screenSize: CGSize
    }

    private static var seeds: [Seed] = []

    convenience init(gpu: FluidGPU, screen: NSScreen) {
        Self.seeds.append(Seed(gpu: gpu, screenSize: screen.frame.size))
        defer { _ = Self.seeds.popLast() }
        self.init(frame: screen.frame, device: gpu.device)
    }

    /// Designated initializer. Implemented so a superclass callback cannot land
    /// in a Swift-synthesized trap. `init(frame:device:)` calls `NSView`'s
    /// `init(frame:)` on super, not on self, so this does not recurse.
    /// `init(frame:)` stays the inherited convenience and forwards here.
    override init(frame frameRect: CGRect, device: MTLDevice?) {
        let resolved = Self.resolve(frame: frameRect)
        gpu = resolved.gpu
        field = resolved.field
        super.init(frame: frameRect, device: resolved.gpu?.device ?? device)
        if resolved.gpu != nil {
            configure()
        }
    }

    required init(coder: NSCoder) {
        let resolved = Self.resolve(frame: .zero)
        gpu = resolved.gpu
        field = resolved.field
        super.init(coder: coder)
        if resolved.gpu != nil {
            configure()
        }
    }

    private static func resolve(frame: CGRect) -> (gpu: FluidGPU?, field: FluidField?) {
        if let seed = seeds.last {
            return (seed.gpu, seed.gpu.makeField(for: seed.screenSize))
        }
        guard let gpu = FluidGPU() else { return (nil, nil) }
        return (gpu, gpu.makeField(for: frame.size))
    }

    private func configure() {
        delegate = self
        isPaused = true
        enableSetNeedsDisplay = false
        colorPixelFormat = .bgra8Unorm
        framebufferOnly = true
        autoResizeDrawable = true
        preferredFramesPerSecond = 60
    }

    func updateGrid(for screenSize: CGSize) {
        guard let gpu else { return }
        let grids = FluidGPU.grids(for: screenSize)
        if grids.velocity.width != field?.velocitySize.width || grids.velocity.height != field?.velocitySize.height {
            field = gpu.makeField(for: screenSize)
        }
    }

    func applyBackground(_ color: SIMD3<Float>) {
        clearColor = MTLClearColor(red: Double(color.x), green: Double(color.y), blue: Double(color.z), alpha: 1)
    }

    func render(
        splats: [Splat],
        obstacles: [Obstacle],
        dt: Float,
        iterations: Int,
        drift: Float,
        background: SIMD3<Float>,
        inks: [SIMD3<Float>],
        intensity: Float
    ) -> Double {
        guard field != nil else { return 0 }
        pending = Pending(
            splats: splats,
            obstacles: obstacles,
            dt: dt,
            iterations: iterations,
            drift: drift,
            background: background,
            inks: inks,
            intensity: intensity
        )
        draw()
        return lastPresentedGPUMilliseconds
    }

    func draw(in view: MTKView) {
        guard let pending, let field else { return }
        lastPresentedGPUMilliseconds = field.stepAndPresent(
            in: view,
            splats: pending.splats,
            obstacles: pending.obstacles,
            dt: pending.dt,
            iterations: pending.iterations,
            drift: pending.drift,
            background: pending.background,
            inks: pending.inks,
            intensity: pending.intensity
        )
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}
