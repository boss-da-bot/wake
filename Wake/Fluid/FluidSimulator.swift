import Metal
import MetalKit
import os
import simd

/// Shared Metal device, shader pipelines, and the per-screen fluid fields.
///
/// Each frame is a classic stable-fluids step: semi-Lagrangian advection,
/// splats from window motion, vorticity confinement, then a Jacobi pressure
/// solve. Velocity stays coarse. Dye and the window mask are six times
/// finer, then drawn with a bicubic sample.
final class FluidGPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let pipelines: [String: MTLComputePipelineState]
    private let renderPipeline: MTLRenderPipelineState

    /// Longest side of the velocity / pressure grid.
    static let velocityMaxDimension = 288
    /// Dye and the obstacle mask are this many velocity cells on a side.
    static let dyeScale = 6

    init?() {
        guard SimLayout.isCompatible() else {
            Log.app.error("Fluid parameter layout does not match the Metal shaders")
            return nil
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary() else {
            Log.app.error("Metal is unavailable")
            return nil
        }
        self.device = device
        self.queue = queue

        let names = [
            "advect", "splat", "curlField", "confine", "divergence",
            "clearField", "jacobi", "gradientSubtract", "rasterObstacles",
            "seedDye", "bloomInk"
        ]
        var compiled: [String: MTLComputePipelineState] = [:]
        compiled.reserveCapacity(names.count)
        do {
            for name in names {
                guard let function = library.makeFunction(name: name) else {
                    Log.fluid.error("Missing Metal function \(name, privacy: .public)")
                    return nil
                }
                compiled[name] = try device.makeComputePipelineState(function: function)
            }
            guard let vertex = library.makeFunction(name: "fullscreenVertex"),
                  let fragment = library.makeFunction(name: "composeFragment") else {
                Log.fluid.error("Missing fullscreen shaders")
                return nil
            }
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = vertex
            descriptor.fragmentFunction = fragment
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            renderPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            Log.fluid.error("Failed to build Metal pipelines: \(error.localizedDescription, privacy: .public)")
            return nil
        }
        pipelines = compiled
    }

    func makeField(for screenSize: CGSize) -> FluidField? {
        let grids = Self.grids(for: screenSize)
        return FluidField(gpu: self, velocityGrid: grids.velocity, dyeGrid: grids.dye)
    }

    struct Grids {
        var velocity: (width: Int, height: Int)
        var dye: (width: Int, height: Int)
    }

    static func grids(for screenSize: CGSize) -> Grids {
        let velocity = sizedGrid(
            screenSize: screenSize,
            longest: velocityMaxDimension,
            minimum: 96
        )
        return Grids(
            velocity: velocity,
            dye: (velocity.width * dyeScale, velocity.height * dyeScale)
        )
    }

    private static func sizedGrid(
        screenSize: CGSize,
        longest: Int,
        minimum: Int
    ) -> (width: Int, height: Int) {
        let width = max(screenSize.width, 1)
        let height = max(screenSize.height, 1)
        let span = CGFloat(longest)
        let gridWidth: Int
        let gridHeight: Int
        if width >= height {
            gridWidth = longest
            gridHeight = Int((span * height / width).rounded())
        } else {
            gridHeight = longest
            gridWidth = Int((span * width / height).rounded())
        }
        return (snap(gridWidth, minimum: minimum), snap(gridHeight, minimum: minimum))
    }

    private static func snap(_ value: Int, minimum: Int) -> Int {
        max(minimum, (value + 7) / 8 * 8)
    }

    fileprivate func pipeline(_ name: String) -> MTLComputePipelineState? {
        pipelines[name]
    }

    fileprivate var composePipeline: MTLRenderPipelineState { renderPipeline }
}

final class FluidField {
    private let gpu: FluidGPU
    let velocitySize: (width: Int, height: Int)
    let dyeSize: (width: Int, height: Int)
    private var velocity: [MTLTexture]
    private var dye: [MTLTexture]
    private var pressure: [MTLTexture]
    private let divergenceTex: MTLTexture
    private let curlTex: MTLTexture
    private let obstacleTex: MTLTexture
    private var velocityIndex = 0
    private var dyeIndex = 0
    private var time: Float = 0
    private var dyeSeeded = false
    private var sinceBloom = Float(0)
    private var bloomRNG: SplitMix
    private var blooms: [LivingBloom] = []
    private var obstacleStamp: UInt64?
    private var previousBuffer: MTLCommandBuffer?

    private enum Tuning {
        static let velocityDecay: Float = 0.85
        static let dyeDecay: Float = 0.022
        static let confinement: Float = 10
    }

    init?(gpu: FluidGPU, velocityGrid: (width: Int, height: Int), dyeGrid: (width: Int, height: Int)) {
        self.gpu = gpu
        velocitySize = velocityGrid
        dyeSize = dyeGrid
        guard let velocity0 = Self.makeTexture(device: gpu.device, width: velocityGrid.width, height: velocityGrid.height, label: "velocity0"),
              let velocity1 = Self.makeTexture(device: gpu.device, width: velocityGrid.width, height: velocityGrid.height, label: "velocity1"),
              let dye0 = Self.makeTexture(device: gpu.device, width: dyeGrid.width, height: dyeGrid.height, label: "dye0"),
              let dye1 = Self.makeTexture(device: gpu.device, width: dyeGrid.width, height: dyeGrid.height, label: "dye1"),
              let pressure0 = Self.makeTexture(device: gpu.device, width: velocityGrid.width, height: velocityGrid.height, label: "pressure0"),
              let pressure1 = Self.makeTexture(device: gpu.device, width: velocityGrid.width, height: velocityGrid.height, label: "pressure1"),
              let divergenceTex = Self.makeTexture(device: gpu.device, width: velocityGrid.width, height: velocityGrid.height, label: "divergence"),
              let curlTex = Self.makeTexture(device: gpu.device, width: velocityGrid.width, height: velocityGrid.height, label: "curl"),
              let obstacleTex = Self.makeTexture(device: gpu.device, width: dyeGrid.width, height: dyeGrid.height, label: "obstacles") else {
            return nil
        }
        velocity = [velocity0, velocity1]
        dye = [dye0, dye1]
        pressure = [pressure0, pressure1]
        self.divergenceTex = divergenceTex
        self.curlTex = curlTex
        self.obstacleTex = obstacleTex
        bloomRNG = SplitMix(seed: UInt64(dyeGrid.width) &+ UInt64(dyeGrid.height) &+ 1)
        clearAll()
    }

    var dyeTexture: MTLTexture { dye[dyeIndex] }

    func stepAndPresent(
        in view: MTKView,
        splats: [Splat],
        obstacles: [Obstacle],
        dt rawDT: Float,
        iterations: Int,
        drift: Float,
        background: SIMD3<Float>,
        inks: [SIMD3<Float>],
        intensity: Float
    ) -> Double {
        guard let commandBuffer = gpu.queue.makeCommandBuffer() else { return 0 }
        commandBuffer.label = "WakeFrame"
        let gpuMilliseconds = takeGPUMilliseconds(replacing: commandBuffer)
        // Idle steps are 1/8 s, and slower on battery. A tighter clamp plays blooms in slow motion.
        let dt = min(max(rawDT, Float(0)), Float(0.25))
        rasterObstacles(commandBuffer, obstacles: obstacles)
        if !dyeSeeded {
            seedDye(commandBuffer, inks: inks, intensity: intensity)
            dyeSeeded = true
        }
        if dt > 0 {
            time += dt
            let velocityDissipation = exp(-Tuning.velocityDecay * dt)
            let dyeDissipation = exp(-Tuning.dyeDecay * dt)
            advect(
                commandBuffer,
                source: velocity[velocityIndex],
                dest: velocity[1 - velocityIndex],
                velocity: velocity[velocityIndex],
                dt: dt,
                dissipation: velocityDissipation
            )
            velocityIndex = 1 - velocityIndex
            advect(
                commandBuffer,
                source: dye[dyeIndex],
                dest: dye[1 - dyeIndex],
                velocity: velocity[velocityIndex],
                dt: dt,
                dissipation: dyeDissipation
            )
            dyeIndex = 1 - dyeIndex

            let limited = Array(splats.prefix(64))
            if !limited.isEmpty {
                applySplat(commandBuffer, source: velocity[velocityIndex], dest: velocity[1 - velocityIndex], splats: limited, mode: 0)
                velocityIndex = 1 - velocityIndex
            }
            let bloomSplats = advanceBlooms(dt: dt, intensity: intensity, inks: inks)
            if !bloomSplats.isEmpty {
                applyBlooms(commandBuffer, source: dye[dyeIndex], dest: dye[1 - dyeIndex], blooms: bloomSplats)
                dyeIndex = 1 - dyeIndex
            }

            dispatch(commandBuffer, "curlField", width: velocitySize.width, height: velocitySize.height) { encoder in
                encoder.setTexture(velocity[velocityIndex], index: 0)
                encoder.setTexture(curlTex, index: 1)
            }
            var forces = ForceParams(
                confinement: Tuning.confinement,
                drift: drift,
                time: time,
                dt: dt
            )
            dispatch(commandBuffer, "confine", width: velocitySize.width, height: velocitySize.height, params: &forces) { encoder in
                encoder.setTexture(velocity[velocityIndex], index: 0)
                encoder.setTexture(curlTex, index: 1)
                encoder.setTexture(velocity[1 - velocityIndex], index: 2)
                encoder.setTexture(obstacleTex, index: 3)
            }
            velocityIndex = 1 - velocityIndex

            dispatch(commandBuffer, "divergence", width: velocitySize.width, height: velocitySize.height) { encoder in
                encoder.setTexture(velocity[velocityIndex], index: 0)
                encoder.setTexture(divergenceTex, index: 1)
                encoder.setTexture(obstacleTex, index: 2)
            }
            dispatch(commandBuffer, "clearField", width: velocitySize.width, height: velocitySize.height) { encoder in
                encoder.setTexture(pressure[0], index: 0)
            }
            var pressureIndex = 0
            let passes = max(1, iterations)
            for _ in 0..<passes {
                dispatch(commandBuffer, "jacobi", width: velocitySize.width, height: velocitySize.height) { encoder in
                    encoder.setTexture(pressure[pressureIndex], index: 0)
                    encoder.setTexture(divergenceTex, index: 1)
                    encoder.setTexture(pressure[1 - pressureIndex], index: 2)
                    encoder.setTexture(obstacleTex, index: 3)
                }
                pressureIndex = 1 - pressureIndex
            }
            dispatch(commandBuffer, "gradientSubtract", width: velocitySize.width, height: velocitySize.height) { encoder in
                encoder.setTexture(velocity[velocityIndex], index: 0)
                encoder.setTexture(pressure[pressureIndex], index: 1)
                encoder.setTexture(velocity[1 - velocityIndex], index: 2)
                encoder.setTexture(obstacleTex, index: 3)
            }
            velocityIndex = 1 - velocityIndex
        }

        if let drawable = view.currentDrawable,
           let pass = view.currentRenderPassDescriptor,
           let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) {
            encoder.setRenderPipelineState(gpu.composePipeline)
            encoder.setFragmentTexture(dye[dyeIndex], index: 0)
            encoder.setFragmentTexture(obstacleTex, index: 1)
            var compose = ComposeParams(background: SIMD4(background.x, background.y, background.z, 1))
            encoder.setFragmentBytes(&compose, length: MemoryLayout<ComposeParams>.stride, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
        return gpuMilliseconds
    }

    /// GPU time of the command buffer submitted on the previous step, if it
    /// has finished. Read on the next frame so nothing waits on the GPU.
    private func takeGPUMilliseconds(replacing buffer: MTLCommandBuffer) -> Double {
        let previous = previousBuffer
        previousBuffer = buffer
        guard let previous, previous.status == .completed else { return 0 }
        let seconds = previous.gpuEndTime - previous.gpuStartTime
        guard seconds.isFinite, seconds > 0 else { return 0 }
        return seconds * 1000
    }

    private func rasterObstacles(_ commandBuffer: MTLCommandBuffer, obstacles: [Obstacle]) {
        let stamp = Self.obstacleStamp(obstacles)
        if stamp == obstacleStamp {
            return
        }
        obstacleStamp = stamp
        let capped = Array(obstacles.prefix(48))
        var list = capped
        if list.isEmpty {
            list = [Obstacle(rect: SIMD4<Float>(0, 0, 0, 0), motion: SIMD4<Float>(0, 0, 0, 0))]
        }
        let header = ObstacleHeader(count: UInt32(capped.count), pad0: 0, pad1: 0, pad2: 0)
        dispatch(commandBuffer, "rasterObstacles", width: dyeSize.width, height: dyeSize.height) { encoder in
            encoder.setTexture(obstacleTex, index: 0)
            list.withUnsafeBytes { raw in
                if let base = raw.baseAddress, raw.count > 0 {
                    encoder.setBytes(base, length: raw.count, index: 0)
                }
            }
            withUnsafeBytes(of: header) { raw in
                if let base = raw.baseAddress {
                    encoder.setBytes(base, length: MemoryLayout<ObstacleHeader>.stride, index: 1)
                }
            }
        }
    }

    private func seedDye(_ commandBuffer: MTLCommandBuffer, inks: [SIMD3<Float>], intensity: Float) {
        let gain = Self.inkGain(intensity)
        var params = SeedParams(
            color0: Self.inkColor(inks, index: 0),
            color1: Self.inkColor(inks, index: 1),
            color2: Self.inkColor(inks, index: 2),
            density: gain,
            phase: bloomRNG.unit() * 6.28318,
            pad0: 0,
            pad1: 0
        )
        dispatch(commandBuffer, "seedDye", width: dyeSize.width, height: dyeSize.height, params: &params) { encoder in
            encoder.setTexture(dye[dyeIndex], index: 0)
            encoder.setTexture(obstacleTex, index: 1)
        }
    }

    private func applyBlooms(
        _ commandBuffer: MTLCommandBuffer,
        source: MTLTexture,
        dest: MTLTexture,
        blooms: [InkBloom]
    ) {
        let list = blooms
        let header = BloomHeader(
            count: UInt32(list.count),
            aspect: Float(dyeSize.width) / Float(max(dyeSize.height, 1)),
            pad0: 0,
            pad1: 0
        )
        dispatch(commandBuffer, "bloomInk", width: dyeSize.width, height: dyeSize.height) { encoder in
            encoder.setTexture(source, index: 0)
            encoder.setTexture(dest, index: 1)
            encoder.setTexture(obstacleTex, index: 2)
            list.withUnsafeBytes { raw in
                if let base = raw.baseAddress, raw.count > 0 {
                    encoder.setBytes(base, length: raw.count, index: 0)
                }
            }
            withUnsafeBytes(of: header) { raw in
                if let base = raw.baseAddress {
                    encoder.setBytes(base, length: MemoryLayout<BloomHeader>.stride, index: 1)
                }
            }
        }
    }

    /// A few soft patches, born one at a time. Fade-in adds dye over several
    /// seconds, the hold tops up what global decay removes, and fade-out
    /// dissolves only that patch.
    private func advanceBlooms(dt: Float, intensity: Float, inks: [SIMD3<Float>]) -> [InkBloom] {
        let clamped = min(1, max(0, intensity))
        let target = 3 + Int((clamped * 5).rounded())
        if blooms.isEmpty {
            for _ in 0..<target {
                let fraction = bloomRNG.unit() * 0.9
                blooms.append(makeBloom(inks: inks, ageFraction: fraction))
            }
            sinceBloom = 0
        }

        sinceBloom += dt
        blooms = blooms.compactMap { bloom in
            var bloom = bloom
            bloom.age += dt
            if bloom.age >= bloom.fadeIn + bloom.hold + bloom.fadeOut {
                return nil
            }
            return bloom
        }
        let interval = 16 - 10 * clamped
        if blooms.count < target, sinceBloom >= interval {
            blooms.append(makeBloom(inks: inks, ageFraction: 0))
            sinceBloom = 0
        }
        if blooms.count > 8 {
            blooms.removeFirst(blooms.count - 8)
        }
        return blooms.map { gpuBloom($0, dt: dt, intensity: clamped) }
    }

    private func makeBloom(inks: [SIMD3<Float>], ageFraction: Float) -> LivingBloom {
        let fadeIn = 5 + bloomRNG.unit() * 3
        let hold = 12 + bloomRNG.unit() * 10
        let fadeOut = 9 + bloomRNG.unit() * 6
        let packed = Self.inkColor(inks, index: Int(bloomRNG.unit() * 3))
        return LivingBloom(
            position: SIMD2(0.18 + bloomRNG.unit() * 0.64, 0.18 + bloomRNG.unit() * 0.64),
            radius: 0.18 + bloomRNG.unit() * 0.14,
            color: SIMD3(packed.x, packed.y, packed.z),
            peak: 0.5 + bloomRNG.unit() * 0.35,
            age: ageFraction * (fadeIn + hold),
            fadeIn: fadeIn,
            hold: hold,
            fadeOut: fadeOut
        )
    }

    private func gpuBloom(_ bloom: LivingBloom, dt: Float, intensity: Float) -> InkBloom {
        let gain = Self.inkGain(intensity)
        let holdEnd = bloom.fadeIn + bloom.hold
        let addRate: Float
        let sustain: Float
        if bloom.age < bloom.fadeIn {
            let t = bloom.age / max(bloom.fadeIn, 0.001)
            let shaped = t * t * (3 - 2 * t)
            addRate = bloom.peak * gain * 2 * shaped / max(bloom.fadeIn, 0.001)
            sustain = 1
        } else if bloom.age < holdEnd {
            addRate = bloom.peak * gain * Tuning.dyeDecay
            sustain = 1
        } else {
            addRate = 0
            let rate = 3 / max(bloom.fadeOut, 0.001)
            sustain = exp(-rate * dt)
        }
        return InkBloom(
            posRadius: SIMD4(bloom.position.x, bloom.position.y, bloom.radius, 0),
            color: SIMD4(bloom.color.x, bloom.color.y, bloom.color.z, addRate * dt),
            fade: SIMD4(sustain, 0, 0, 0)
        )
    }

    private static func inkGain(_ intensity: Float) -> Float {
        0.4 + 0.6 * min(1, max(0, intensity))
    }

    private static func inkColor(_ inks: [SIMD3<Float>], index: Int) -> SIMD4<Float> {
        let color = inks.isEmpty ? SIMD3<Float>(0.72, 0.74, 0.78) : inks[index % inks.count]
        return SIMD4(color.x, color.y, color.z, 1)
    }

    private func advect(
        _ commandBuffer: MTLCommandBuffer,
        source: MTLTexture,
        dest: MTLTexture,
        velocity velocityTexture: MTLTexture,
        dt: Float,
        dissipation: Float
    ) {
        var params = AdvectParams(dt: dt, dissipation: dissipation, pad0: 0, pad1: 0)
        dispatch(commandBuffer, "advect", width: dest.width, height: dest.height, params: &params) { encoder in
            encoder.setTexture(source, index: 0)
            encoder.setTexture(dest, index: 1)
            encoder.setTexture(velocityTexture, index: 2)
            encoder.setTexture(obstacleTex, index: 3)
        }
    }

    private func applySplat(
        _ commandBuffer: MTLCommandBuffer,
        source: MTLTexture,
        dest: MTLTexture,
        splats: [Splat],
        mode: UInt32
    ) {
        let aspect = Float(velocitySize.width) / Float(max(velocitySize.height, 1))
        let header = SplatHeader(count: UInt32(splats.count), mode: mode, aspect: aspect, pad: 0)
        let copy = splats
        dispatch(commandBuffer, "splat", width: velocitySize.width, height: velocitySize.height) { encoder in
            encoder.setTexture(source, index: 0)
            encoder.setTexture(dest, index: 1)
            encoder.setTexture(obstacleTex, index: 2)
            copy.withUnsafeBytes { raw in
                if let base = raw.baseAddress, raw.count > 0 {
                    encoder.setBytes(base, length: raw.count, index: 0)
                }
            }
            withUnsafeBytes(of: header) { raw in
                if let base = raw.baseAddress {
                    encoder.setBytes(base, length: MemoryLayout<SplatHeader>.stride, index: 1)
                }
            }
        }
    }

    private func dispatch(
        _ commandBuffer: MTLCommandBuffer,
        _ name: String,
        width gridWidth: Int,
        height gridHeight: Int,
        _ configure: (MTLComputeCommandEncoder) -> Void
    ) {
        guard let pipeline = gpu.pipeline(name),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = name
        encoder.setComputePipelineState(pipeline)
        configure(encoder)
        let width = max(pipeline.threadExecutionWidth, 1)
        let budget = max(pipeline.maxTotalThreadsPerThreadgroup, width)
        let height = max(1, min(16, budget / width))
        let threads = MTLSize(width: width, height: height, depth: 1)
        let groups = MTLSize(
            width: (gridWidth + threads.width - 1) / threads.width,
            height: (gridHeight + threads.height - 1) / threads.height,
            depth: 1
        )
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }

    private func dispatch<T>(
        _ commandBuffer: MTLCommandBuffer,
        _ name: String,
        width gridWidth: Int,
        height gridHeight: Int,
        params: inout T,
        _ configure: (MTLComputeCommandEncoder) -> Void
    ) {
        let bytes = params
        dispatch(commandBuffer, name, width: gridWidth, height: gridHeight) { encoder in
            withUnsafeBytes(of: bytes) { raw in
                if let base = raw.baseAddress {
                    encoder.setBytes(base, length: MemoryLayout<T>.stride, index: 0)
                }
            }
            configure(encoder)
        }
    }

    /// Stable while a window is still. Motion is quantized so a resting rect
    /// does not re-raster the mask on every idle frame.
    private static func obstacleStamp(_ obstacles: [Obstacle]) -> UInt64 {
        var hash: UInt64 = 0x6A09E667F3BCC909
        hash ^= UInt64(min(obstacles.count, 48))
        for obstacle in obstacles.prefix(48) {
            hash = mixStamp(hash, quantize(obstacle.rect.x, scale: 8192))
            hash = mixStamp(hash, quantize(obstacle.rect.y, scale: 8192))
            hash = mixStamp(hash, quantize(obstacle.rect.z, scale: 8192))
            hash = mixStamp(hash, quantize(obstacle.rect.w, scale: 8192))
            hash = mixStamp(hash, quantize(obstacle.motion.x, scale: 512))
            hash = mixStamp(hash, quantize(obstacle.motion.y, scale: 512))
            hash = mixStamp(hash, quantize(obstacle.motion.z, scale: 512))
            hash = mixStamp(hash, quantize(obstacle.motion.w, scale: 512))
        }
        return hash
    }

    private static func quantize(_ value: Float, scale: Float) -> UInt64 {
        let rounded = (value * scale).rounded()
        return UInt64(bitPattern: Int64(rounded))
    }

    private static func mixStamp(_ hash: UInt64, _ value: UInt64) -> UInt64 {
        var mixed = hash ^ (value &+ 0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> 30)) &* 0xBF58476D1CE4E5B9
        return mixed ^ (mixed >> 27)
    }

    private func clearAll() {
        guard let commandBuffer = gpu.queue.makeCommandBuffer() else { return }
        let textures = velocity + dye + pressure + [obstacleTex]
        for texture in textures {
            dispatch(commandBuffer, "clearField", width: texture.width, height: texture.height) { encoder in
                encoder.setTexture(texture, index: 0)
            }
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    private static func makeTexture(device: MTLDevice, width: Int, height: Int, label: String) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        let texture = device.makeTexture(descriptor: descriptor)
        texture?.label = label
        return texture
    }
}

private struct LivingBloom {
    var position: SIMD2<Float>
    var radius: Float
    var color: SIMD3<Float>
    var peak: Float
    var age: Float
    var fadeIn: Float
    var hold: Float
    var fadeOut: Float
}

private struct SplitMix {
    var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0xA24BAED4963EE407 : seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func unit() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}
