import Foundation
import IOKit.ps

/// Coarse power signals used to drop the frame rate. Polled, not subscribed,
/// so a desktop Mac without a battery just stays on the plugged-in path.
final class PowerMonitor {
    private(set) var isOnBattery = false
    private(set) var isLowPowerMode = false
    private(set) var isThermalLimited = false
    private var lastRefresh = Date.distantPast

    func refreshIfNeeded(force: Bool = false) {
        if !force && Date().timeIntervalSince(lastRefresh) < 5 {
            return
        }
        lastRefresh = Date()
        isLowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
        let thermal = ProcessInfo.processInfo.thermalState
        isThermalLimited = thermal == .serious || thermal == .critical
        isOnBattery = Self.readOnBattery()
    }

    var prefersReducedRate: Bool {
        isOnBattery || isLowPowerMode || isThermalLimited
    }

    private static func readOnBattery() -> Bool {
        guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue() else {
            return false
        }
        guard let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [CFTypeRef] else {
            return false
        }

        let stateKey = kIOPSPowerSourceStateKey as String
        let batteryValue = kIOPSBatteryPowerValue as String
        let acValue = kIOPSACPowerValue as String
        var sawBattery = false
        var sawAC = false

        for source in sources {
            guard let description = IOPSGetPowerSourceDescription(snapshot, source)?.takeUnretainedValue() as? [String: Any] else {
                continue
            }
            let state = description[stateKey] as? String
            if state == batteryValue {
                sawBattery = true
            } else if state == acValue {
                sawAC = true
            }
        }
        return sawBattery && !sawAC
    }
}
