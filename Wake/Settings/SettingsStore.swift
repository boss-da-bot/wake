import Foundation

/// The only data Wake writes: palette, intensity, and pause, in UserDefaults.
final class SettingsStore {
    private let defaults: UserDefaults

    private enum Key {
        static let palette = "palette"
        static let intensity = "intensity"
        static let paused = "paused"
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    var paletteID: String {
        get { defaults.string(forKey: Key.palette) ?? Palette.ink.id }
        set { defaults.set(newValue, forKey: Key.palette) }
    }

    var intensity: Double {
        get {
            guard defaults.object(forKey: Key.intensity) != nil else { return 0.65 }
            return min(1, max(0, defaults.double(forKey: Key.intensity)))
        }
        set { defaults.set(min(1, max(0, newValue)), forKey: Key.intensity) }
    }

    var isPaused: Bool {
        get { defaults.bool(forKey: Key.paused) }
        set { defaults.set(newValue, forKey: Key.paused) }
    }
}
