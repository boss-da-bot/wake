import AppKit
import MetalKit
import simd

final class FluidView: MTKView, MTKViewDelegate {
    private let gpu: FluidGPU
    private var field: FluidField?
    private var pending: Pending?

    private struct Pending {
        var splats: [Splat]
        var dt: Float
        var iterations: Int
        var drift: Float
        var background: SIMD3<Float>
    }

    init(gpu: FluidGPU, screen: NSScreen) {
        self.gpu = gpu
        field = gpu.makeField(for: screen.frame.size)
        super.init(frame: screen.frame, device: gpu.device)
        delegate = self
        isPaused = true
        enableSetNeedsDisplay = false
        colorPixelFormat = .bgra8Unorm
        framebufferOnly = true
        autoResizeDrawable = true
        preferredFramesPerSecond = 60
    }

    @available(*, unavailable)
    required init(coder: NSCoder) {
        fatalError("Wake wallpaper views are created in code")
    }

    func updateGrid(for screenSize: CGSize) {
        let grid = FluidGPU.gridSize(for: screenSize)
        if grid.width != field?.gridSize.width || grid.height != field?.gridSize.height {
            field = gpu.makeField(for: screenSize)
        }
    }

    func applyBackground(_ color: SIMD3<Float>) {
        clearColor = MTLClearColor(red: Double(color.x), green: Double(color.y), blue: Double(color.z), alpha: 1)
    }

    func render(splats: [Splat], dt: Float, iterations: Int, drift: Float, background: SIMD3<Float>) {
        guard field != nil else { return }
        pending = Pending(splats: splats, dt: dt, iterations: iterations, drift: drift, background: background)
        draw()
    }

    func draw(in view: MTKView) {
        guard let pending, let field else { return }
        field.stepAndPresent(
            in: view,
            splats: pending.splats,
            dt: pending.dt,
            iterations: pending.iterations,
            drift: pending.drift,
            background: pending.background
        )
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}
