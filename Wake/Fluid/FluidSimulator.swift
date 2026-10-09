import Metal
import MetalKit
import os
import simd

/// Shared Metal device, shader pipelines, and the per-screen fluid fields.
///
/// Each frame is a classic stable-fluids step: semi-Lagrangian advection,
/// splats from window motion, vorticity confinement, then a Jacobi pressure
/// solve. The grid stays small (longest side 320) so a fullscreen drawable
/// only upscales a bilinear ink texture.
final class FluidGPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let pipelines: [String: MTLComputePipelineState]
    private let renderPipeline: MTLRenderPipelineState

    static let maxDimension = 320

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
            "clearField", "jacobi", "gradientSubtract"
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
        let grid = Self.gridSize(for: screenSize)
        return FluidField(gpu: self, width: grid.width, height: grid.height)
    }

    static func gridSize(for screenSize: CGSize) -> (width: Int, height: Int) {
        let width = max(screenSize.width, 1)
        let height = max(screenSize.height, 1)
        let longest = CGFloat(maxDimension)
        let gridWidth: Int
        let gridHeight: Int
        if width >= height {
            gridWidth = maxDimension
            gridHeight = Int((longest * height / width).rounded())
        } else {
            gridHeight = maxDimension
            gridWidth = Int((longest * width / height).rounded())
        }
        return (snap(gridWidth), snap(gridHeight))
    }

    private static func snap(_ value: Int) -> Int {
        max(96, (value + 7) / 8 * 8)
    }

    fileprivate func pipeline(_ name: String) -> MTLComputePipelineState? {
        pipelines[name]
    }

    fileprivate var composePipeline: MTLRenderPipelineState { renderPipeline }
}

final class FluidField {
    private let gpu: FluidGPU
    let gridSize: (width: Int, height: Int)
    private var velocity: [MTLTexture]
    private var dye: [MTLTexture]
    private var pressure: [MTLTexture]
    private let divergenceTex: MTLTexture
    private let curlTex: MTLTexture
    private var velocityIndex = 0
    private var dyeIndex = 0
    private var time: Float = 0

    private enum Tuning {
        static let velocityDecay: Float = 0.85
        static let dyeDecay: Float = 0.12
        static let confinement: Float = 10
    }

    init?(gpu: FluidGPU, width: Int, height: Int) {
        self.gpu = gpu
        gridSize = (width, height)
        guard let velocity0 = Self.makeTexture(device: gpu.device, width: width, height: height, label: "velocity0"),
              let velocity1 = Self.makeTexture(device: gpu.device, width: width, height: height, label: "velocity1"),
              let dye0 = Self.makeTexture(device: gpu.device, width: width, height: height, label: "dye0"),
              let dye1 = Self.makeTexture(device: gpu.device, width: width, height: height, label: "dye1"),
              let pressure0 = Self.makeTexture(device: gpu.device, width: width, height: height, label: "pressure0"),
              let pressure1 = Self.makeTexture(device: gpu.device, width: width, height: height, label: "pressure1"),
              let divergenceTex = Self.makeTexture(device: gpu.device, width: width, height: height, label: "divergence"),
              let curlTex = Self.makeTexture(device: gpu.device, width: width, height: height, label: "curl") else {
            return nil
        }
        velocity = [velocity0, velocity1]
        dye = [dye0, dye1]
        pressure = [pressure0, pressure1]
        self.divergenceTex = divergenceTex
        self.curlTex = curlTex
        clearAll()
    }

    var dyeTexture: MTLTexture { dye[dyeIndex] }

    func stepAndPresent(
        in view: MTKView,
        splats: [Splat],
        dt rawDT: Float,
        iterations: Int,
        drift: Float,
        background: SIMD3<Float>
    ) {
        guard let commandBuffer = gpu.queue.makeCommandBuffer() else { return }
        commandBuffer.label = "WakeFrame"
        let dt = min(max(rawDT, Float(0)), Float(0.05))
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
                applySplat(commandBuffer, source: dye[dyeIndex], dest: dye[1 - dyeIndex], splats: limited, mode: 1)
                dyeIndex = 1 - dyeIndex
            }

            dispatch(commandBuffer, "curlField") { encoder in
                encoder.setTexture(velocity[velocityIndex], index: 0)
                encoder.setTexture(curlTex, index: 1)
            }
            var forces = ForceParams(
                confinement: Tuning.confinement,
                drift: drift,
                time: time,
                dt: dt
            )
            dispatch(commandBuffer, "confine", params: &forces) { encoder in
                encoder.setTexture(velocity[velocityIndex], index: 0)
                encoder.setTexture(curlTex, index: 1)
                encoder.setTexture(velocity[1 - velocityIndex], index: 2)
            }
            velocityIndex = 1 - velocityIndex

            dispatch(commandBuffer, "divergence") { encoder in
                encoder.setTexture(velocity[velocityIndex], index: 0)
                encoder.setTexture(divergenceTex, index: 1)
            }
            dispatch(commandBuffer, "clearField") { encoder in
                encoder.setTexture(pressure[0], index: 0)
            }
            var pressureIndex = 0
            let passes = max(1, iterations)
            for _ in 0..<passes {
                dispatch(commandBuffer, "jacobi") { encoder in
                    encoder.setTexture(pressure[pressureIndex], index: 0)
                    encoder.setTexture(divergenceTex, index: 1)
                    encoder.setTexture(pressure[1 - pressureIndex], index: 2)
                }
                pressureIndex = 1 - pressureIndex
            }
            dispatch(commandBuffer, "gradientSubtract") { encoder in
                encoder.setTexture(velocity[velocityIndex], index: 0)
                encoder.setTexture(pressure[pressureIndex], index: 1)
                encoder.setTexture(velocity[1 - velocityIndex], index: 2)
            }
            velocityIndex = 1 - velocityIndex
        }

        if let drawable = view.currentDrawable,
           let pass = view.currentRenderPassDescriptor,
           let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) {
            encoder.setRenderPipelineState(gpu.composePipeline)
            encoder.setFragmentTexture(dye[dyeIndex], index: 0)
            var compose = ComposeParams(background: SIMD4(background.x, background.y, background.z, 1))
            encoder.setFragmentBytes(&compose, length: MemoryLayout<ComposeParams>.stride, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
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
        dispatch(commandBuffer, "advect", params: &params) { encoder in
            encoder.setTexture(source, index: 0)
            encoder.setTexture(dest, index: 1)
            encoder.setTexture(velocityTexture, index: 2)
        }
    }

    private func applySplat(
        _ commandBuffer: MTLCommandBuffer,
        source: MTLTexture,
        dest: MTLTexture,
        splats: [Splat],
        mode: UInt32
    ) {
        let aspect = Float(gridSize.width) / Float(max(gridSize.height, 1))
        let header = SplatHeader(count: UInt32(splats.count), mode: mode, aspect: aspect, pad: 0)
        let copy = splats
        dispatch(commandBuffer, "splat") { encoder in
            encoder.setTexture(source, index: 0)
            encoder.setTexture(dest, index: 1)
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
            width: (gridSize.width + threads.width - 1) / threads.width,
            height: (gridSize.height + threads.height - 1) / threads.height,
            depth: 1
        )
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }

    private func dispatch<T>(
        _ commandBuffer: MTLCommandBuffer,
        _ name: String,
        params: inout T,
        _ configure: (MTLComputeCommandEncoder) -> Void
    ) {
        let bytes = params
        dispatch(commandBuffer, name) { encoder in
            withUnsafeBytes(of: bytes) { raw in
                if let base = raw.baseAddress {
                    encoder.setBytes(base, length: MemoryLayout<T>.stride, index: 0)
                }
            }
            configure(encoder)
        }
    }

    private func clearAll() {
        guard let commandBuffer = gpu.queue.makeCommandBuffer() else { return }
        let textures = velocity + dye + pressure
        for texture in textures {
            dispatch(commandBuffer, "clearField") { encoder in
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
