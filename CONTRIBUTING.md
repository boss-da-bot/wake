# Contributing

Wake is a small macOS app. Changes should stay light, offline, and easy to build.

## Build

You need Xcode 15 or later on macOS 13 or later.

```bash
./Scripts/build.sh
open build/Build/Products/Release/Wake.app
```

The script produces an ad-hoc signed app when Xcode will sign with `-`, or an unsigned app when signing is disabled. Either one runs from a local build. A downloaded copy needs the right-click Open step described in the README.

## What to keep true

- No network access, telemetry, or accounts.
- Do not request Screen Recording or Accessibility. Window stirring uses bounds from `CGWindowListCopyWindowInfo` only.
- Do not store anything except the existing UserDefaults settings (palette, intensity, pause). Launch at login is handled by the system.
- Prefer no third-party dependencies. If you add one, it must use a permissive license (MIT, BSD, Apache-2.0, or similar) and be credited in the README.
- Keep the simulation grid small and the frame cap at 60. Idle, battery, and covered-desktop slowdowns are part of the product.

## Fluid tuning

Motion gains live in `WindowMotionTracker.Tuning`. Decay, confinement, and grid size live in `FluidSimulator.swift` and `FluidGPU`. The Metal pipeline is in `Wake/Fluid/Shaders.metal`.

## Checks

GitHub Actions builds a universal (arm64 + x86_64) Release app on macOS and uploads `Wake-unsigned.zip`. A green run on your pull request is the compile check. This repository's cloud environment is Linux, so don't expect `swift build` to work there.

## License

By contributing, you agree that your changes are licensed under the MIT License in `LICENSE`.
