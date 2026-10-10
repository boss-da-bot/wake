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

/// One solid window in uv space. `rect` is min.xy, max.zw.
/// `motion` is origin velocity xy and size velocity zw, both in uv/s.
struct Obstacle {
    var rect: SIMD4<Float>
    var motion: SIMD4<Float>
}

struct ObstacleHeader {
    var count: UInt32
    var pad0: UInt32
    var pad1: Float
    var pad2: Float
}

struct SeedParams {
    var color0: SIMD4<Float>
    var color1: SIMD4<Float>
    var color2: SIMD4<Float>
    var density: Float
    var phase: Float
    var pad0: Float
    var pad1: Float
}

/// One living ink patch. `color.w` is dye added this frame. `fade.x` is the
/// center multiplier for a patch that is dissolving (1 leaves dye alone).
struct InkBloom {
    var posRadius: SIMD4<Float>
    var color: SIMD4<Float>
    var fade: SIMD4<Float>
}

struct BloomHeader {
    var count: UInt32
    var aspect: Float
    var pad0: Float
    var pad1: Float
}

enum SimLayout {
    /// Mismatched Swift and Metal layouts disable the wallpaper instead of trapping at launch.
    static func isCompatible() -> Bool {
        MemoryLayout<Splat>.stride == 48
            && MemoryLayout<Splat>.alignment == 16
            && MemoryLayout<AdvectParams>.stride == 16
            && MemoryLayout<ForceParams>.stride == 16
            && MemoryLayout<SplatHeader>.stride == 16
            && MemoryLayout<ComposeParams>.stride == 16
            && MemoryLayout<Obstacle>.stride == 32
            && MemoryLayout<Obstacle>.alignment == 16
            && MemoryLayout<ObstacleHeader>.stride == 16
            && MemoryLayout<SeedParams>.stride == 64
            && MemoryLayout<SeedParams>.alignment == 16
            && MemoryLayout<InkBloom>.stride == 48
            && MemoryLayout<InkBloom>.alignment == 16
            && MemoryLayout<BloomHeader>.stride == 16
    }
}
