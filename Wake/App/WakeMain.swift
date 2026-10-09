import AppKit
import os

enum Log {
    static let app = Logger(subsystem: "app.wake.desktop", category: "app")
    static let fluid = Logger(subsystem: "app.wake.desktop", category: "fluid")
}

@main
enum WakeMain {
    static let delegate = AppDelegate()

    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        app.delegate = delegate
        app.run()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var model: AppModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        let model = AppModel()
        model.start()
        self.model = model
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }
}
