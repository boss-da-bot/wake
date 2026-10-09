import simd

/// GPU splat. Layout is shared with `Splat` in Shaders.metal (three float4s).
struct Splat {
    var posRadius: SIMD4<Float>
    var velocity: SIMD4<Float>
    var color: SIMD4<Float>

    init(position: SIMD2<Float>, radius: Float, impulse: SIMD2<Float>, color: SIMD3<Float>, amount: Float) {
        posRadius = SIMD4(position.x, position.y, radius, 0)
        velocity = SIMD4(impulse.x, impulse.y, 0, 0)
        self.color = SIMD4(color.x, color.y, color.z, amount)
    }
}

struct AdvectParams {
    var dt: Float
    var dissipation: Float
    var pad0: Float
    var pad1: Float
}

struct ForceParams {
    var confinement: Float
    var drift: Float
    var time: Float
    var dt: Float
}

struct SplatHeader {
    var count: UInt32
    var mode: UInt32
    var aspect: Float
    var pad: Float
}

struct ComposeParams {
    var background: SIMD4<Float>
}

enum SimLayout {
    static func assertAligned() {
        precondition(MemoryLayout<Splat>.stride == 48, "Splat stride must match the Metal struct")
        precondition(MemoryLayout<Splat>.alignment == 16, "Splat alignment must match the Metal struct")
        precondition(MemoryLayout<AdvectParams>.stride == 16, "AdvectParams stride must match the Metal struct")
        precondition(MemoryLayout<ForceParams>.stride == 16, "ForceParams stride must match the Metal struct")
        precondition(MemoryLayout<SplatHeader>.stride == 16, "SplatHeader stride must match the Metal struct")
        precondition(MemoryLayout<ComposeParams>.stride == 16, "ComposeParams stride must match the Metal struct")
    }
}
