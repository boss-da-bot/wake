import AppKit
import ServiceManagement

final class MenuBarController: NSObject, NSMenuDelegate {
    private weak var model: AppModel?
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
    private let pauseItem = NSMenuItem()
    private let statsItem = NSMenuItem(title: "Measuring…", action: nil, keyEquivalent: "")
    private let loginItem = NSMenuItem()
    private let intensitySlider = NSSlider(value: 0.65, minValue: 0, maxValue: 1, target: nil, action: nil)
    private let percentField = NSTextField(labelWithString: "65%")
    private var paletteItems: [NSMenuItem] = []

    func install(model: AppModel) {
        self.model = model
        intensitySlider.doubleValue = model.intensity
        intensitySlider.isContinuous = true
        intensitySlider.target = self
        intensitySlider.action = #selector(intensityChanged(_:))
        percentField.stringValue = Self.percentString(model.intensity)

        let menu = NSMenu()
        menu.autoenablesItems = false
        menu.delegate = self

        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0"
        let header = NSMenuItem(title: "Wake \(version)", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)

        let hint = NSMenuItem(title: "Drag a window to stir the ink", action: nil, keyEquivalent: "")
        hint.isEnabled = false
        menu.addItem(hint)

        statsItem.isEnabled = false
        statsItem.isHidden = true
        menu.addItem(statsItem)

        if !model.metalAvailable {
            let unavailable = NSMenuItem(title: "Metal is unavailable", action: nil, keyEquivalent: "")
            unavailable.isEnabled = false
            menu.addItem(unavailable)
        }

        menu.addItem(.separator())

        pauseItem.target = self
        pauseItem.action = #selector(togglePause(_:))
        menu.addItem(pauseItem)

        menu.addItem(.separator())

        let paletteMenu = NSMenu()
        paletteMenu.autoenablesItems = false
        for palette in Palette.all {
            let item = NSMenuItem(title: palette.name, action: #selector(selectPalette(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = palette.id
            item.isEnabled = true
            paletteMenu.addItem(item)
            paletteItems.append(item)
        }
        let paletteRoot = NSMenuItem(title: "Palette", action: nil, keyEquivalent: "")
        paletteRoot.submenu = paletteMenu
        paletteRoot.isEnabled = true
        menu.addItem(paletteRoot)

        menu.addItem(.separator())
        let intensityItem = NSMenuItem()
        intensityItem.view = makeIntensityView()
        menu.addItem(intensityItem)

        menu.addItem(.separator())
        loginItem.target = self
        loginItem.action = #selector(toggleLogin(_:))
        loginItem.isEnabled = true
        menu.addItem(loginItem)

        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Wake", action: #selector(quit(_:)), keyEquivalent: "q")
        quit.target = self
        quit.isEnabled = true
        menu.addItem(quit)

        statusItem.menu = menu
        applyPaused(model.isPaused)
        refresh()
    }

    func applyPaused(_ paused: Bool) {
        let symbol = paused ? "drop" : "drop.fill"
        let image = NSImage(systemSymbolName: symbol, accessibilityDescription: "Wake")
        let configured = image?.withSymbolConfiguration(
            NSImage.SymbolConfiguration(pointSize: 15, weight: .medium)
        )
        configured?.isTemplate = true
        if let configured {
            statusItem.button?.image = configured
        } else {
            statusItem.button?.title = "W"
        }
        statusItem.button?.toolTip = paused ? "Wake (paused)" : "Wake"
        pauseItem.title = paused ? "Resume" : "Pause"
    }

    func menuWillOpen(_ menu: NSMenu) {
        if NSEvent.modifierFlags.contains(.option) {
            model?.toggleFrameStats()
        }
        refresh()
    }

    func refresh() {
        guard let model else { return }
        applyPaused(model.isPaused)
        let showingStats = model.showsFrameStats
        statsItem.isHidden = !showingStats
        if showingStats {
            statsItem.title = model.frameStatsLine.isEmpty ? "Measuring…" : model.frameStatsLine
        }
        for item in paletteItems {
            let id = item.representedObject as? String
            item.state = id == model.palette.id ? .on : .off
        }
        intensitySlider.doubleValue = model.intensity
        percentField.stringValue = Self.percentString(model.intensity)

        switch SMAppService.mainApp.status {
        case .enabled:
            loginItem.title = "Launch at Login"
            loginItem.state = .on
        case .requiresApproval:
            loginItem.title = "Launch at Login (Needs Approval)"
            loginItem.state = .mixed
        default:
            loginItem.title = "Launch at Login"
            loginItem.state = .off
        }
    }

    private func makeIntensityView() -> NSView {
        let container = NSView(frame: NSRect(x: 0, y: 0, width: 240, height: 48))
        let caption = NSTextField(labelWithString: "Intensity")
        caption.font = NSFont.menuFont(ofSize: 13)
        caption.textColor = .labelColor
        caption.frame = NSRect(x: 18, y: 26, width: 120, height: 16)
        percentField.font = NSFont.menuFont(ofSize: 13)
        percentField.textColor = .secondaryLabelColor
        percentField.alignment = .right
        percentField.frame = NSRect(x: 140, y: 26, width: 80, height: 16)
        intensitySlider.frame = NSRect(x: 18, y: 4, width: 204, height: 22)
        container.addSubview(caption)
        container.addSubview(percentField)
        container.addSubview(intensitySlider)
        return container
    }

    @objc private func togglePause(_ sender: Any?) {
        model?.togglePaused()
    }

    @objc private func selectPalette(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String else { return }
        model?.setPalette(Palette.named(id))
        refresh()
    }

    @objc private func intensityChanged(_ sender: NSSlider) {
        model?.setIntensity(sender.doubleValue)
        percentField.stringValue = Self.percentString(sender.doubleValue)
    }

    @objc private func toggleLogin(_ sender: Any?) {
        do {
            let service = SMAppService.mainApp
            if service.status == .enabled {
                try service.unregister()
            } else {
                try service.register()
            }
        } catch {
            presentLoginError(error)
        }
        refresh()
    }

    @objc private func quit(_ sender: Any?) {
        NSApp.terminate(nil)
    }

    private func presentLoginError(_ error: Error) {
        if #available(macOS 14.0, *) {
            NSApp.activate()
        } else {
            NSApp.activate(ignoringOtherApps: true)
        }
        let alert = NSAlert()
        alert.messageText = "Couldn't change launch at login"
        alert.informativeText = error.localizedDescription
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    private static func percentString(_ value: Double) -> String {
        "\(Int((value * 100).rounded()))%"
    }
}
