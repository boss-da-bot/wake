import simd

/// Ink colors stirred into the simulation. Backgrounds stay near-black so the
/// desktop reads as a dark wallpaper between motions.
struct Palette: Equatable, Identifiable {
    var id: String
    var name: String
    var background: SIMD3<Float>
    var inks: [SIMD3<Float>]

    static let ink = Palette(
        id: "ink",
        name: "Ink",
        background: SIMD3(0.012, 0.014, 0.020),
        inks: [
            SIMD3(0.62, 0.70, 0.86),
            SIMD3(0.30, 0.38, 0.64),
            SIMD3(0.78, 0.82, 0.88)
        ]
    )

    static let ember = Palette(
        id: "ember",
        name: "Ember",
        background: SIMD3(0.020, 0.012, 0.010),
        inks: [
            SIMD3(0.95, 0.46, 0.16),
            SIMD3(0.72, 0.18, 0.12),
            SIMD3(1.00, 0.74, 0.38)
        ]
    )

    static let aurora = Palette(
        id: "aurora",
        name: "Aurora",
        background: SIMD3(0.010, 0.016, 0.018),
        inks: [
            SIMD3(0.28, 0.86, 0.68),
            SIMD3(0.46, 0.36, 0.95),
            SIMD3(0.38, 0.72, 0.96)
        ]
    )

    static let mono = Palette(
        id: "mono",
        name: "Mono",
        background: SIMD3(0.012, 0.012, 0.013),
        inks: [
            SIMD3(0.84, 0.84, 0.86),
            SIMD3(0.46, 0.46, 0.50),
            SIMD3(0.70, 0.70, 0.72)
        ]
    )

    static let all: [Palette] = [.ink, .ember, .aurora, .mono]

    static func named(_ id: String) -> Palette {
        all.first { $0.id == id } ?? .ink
    }

    func color(forWindow id: UInt32) -> SIMD3<Float> {
        let index = Int(id % UInt32(inks.count))
        let base = inks[index]
        let highlight = inks[(index + 1) % inks.count]
        return base * 0.78 + highlight * 0.22
    }
}
