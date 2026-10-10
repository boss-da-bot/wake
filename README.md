# Wake

Ink that moves when your windows do.

Wake is a free, open-source menu bar app for macOS. It draws a dark fluid simulation behind your windows, under the desktop icons. The desktop starts full of ink. Patches bloom and fade on their own, slowly and not all at once. A window is a solid in that pool: dragging it only stirs the ink that is already there.

![Wake placeholder](docs/demo.gif)

The image above is a **placeholder**, not a screen recording. Replace `docs/demo.gif` with a capture of a real window drag when you have one.

[![Build](https://github.com/boss-da-bot/wake/actions/workflows/build.yml/badge.svg)](https://github.com/boss-da-bot/wake/actions/workflows/build.yml)

## Install

Wake is unsigned. macOS will refuse a normal double-click the first time you open a downloaded copy.

1. Download `Wake-unsigned.zip` from the [latest build artifact](https://github.com/boss-da-bot/wake/actions/workflows/build.yml) (or from a [release](https://github.com/boss-da-bot/wake/releases) if one has been tagged).
2. Unzip it and, if you like, move `Wake.app` to `/Applications`.
3. **Right-click `Wake.app` and choose Open**, then Open again in the dialog.

If that dialog never appears, open **System Settings → Privacy & Security** and click **Open Anyway** next to Wake.

The app has no Dock icon. Look for the drop in the menu bar.

From Terminal, the same Gatekeeper exception is:

```bash
xattr -dr com.apple.quarantine /Applications/Wake.app
open /Applications/Wake.app
```

## Menu

- **Pause / Resume**
- **Palette** — Ink, Ember, Aurora, Mono
- **Intensity**
- **Launch at Login**
- **Quit**

Drag a window across the desktop to stir the ink. Intensity sets how dense the ink is and how often a new patch appears. Nothing here is configurable beyond that menu, on purpose.

Hold **Option** while opening the menu to show a line with the recent frame rate and GPU time, summed across displays. While that line is visible, the same numbers are written about every two seconds to the unified log:

```bash
log stream --predicate 'subsystem == "app.wake.desktop"' --level info
```

## How it works

Wake is a [stable fluids](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf) simulation (Jos Stam, SIGGRAPH 1999) with [vorticity confinement](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/smoke.pdf) (Fedkiw, Stam, and Jensen, SIGGRAPH 2001), running on the GPU in Metal. Each frame advects velocity and dye, adds forces, solves pressure with a short Jacobi loop, and draws the dye into a borderless window. Window rectangles are rasterized into an obstacle mask: velocity and dye inside them are cleared, and the pressure solve treats their edges as moving walls.

That window sits just under the desktop icons, on every display and Space, and it ignores clicks. It is the wallpaper, not an overlay.

The part other fluid wallpapers skip: each frame Wake reads **on-screen window bounds** with `CGWindowListCopyWindowInfo` and diffs them. The rectangle is a solid, so the area a window just left starts empty and neighboring ink is drawn back in. The leading edge pushes fluid ahead; the trailing edge leaves a wake. Opening, closing, resizing, and the pointer only add velocity, never new ink. The pointer is sampled with `NSEvent.mouseLocation`, which does not need Accessibility access.

Velocity and pressure use a coarse grid (long side 288). Ink and the window mask are six times finer, advected with a smooth sample of that velocity, then drawn with a bicubic filter and a little dither so the wallpaper stays soft on a Retina or 5K display. Each display has its own field, because the aspect ratios differ. They share one Metal device.

While a window is moving, Wake runs at 60 fps (30 on battery, 20 when thermally limited or in Low Power Mode) and reads the window list every frame. After the motion settles it drops to about 8 fps, reads the window list about twice a second, and shortens the pressure solve. A covered display, Pause, or display sleep stops rendering and leaves a one-hertz timer. A drag can take up to half a second to be noticed from idle, then it is 60 fps.

These are design targets for an Apple silicon Mac with a built-in Retina display and a 5K ultrawide. This repository cannot measure them.

- **Idle**, blooms drifting: well under 2–3% CPU. GPU time should be a fraction of a millisecond, plus the cost of presenting two framebuffers at about 8 fps.
- **Dragging a window**: 60 fps. CPU rises because the window list is read every frame. GPU time rises with the finer ink grid and a full pressure solve, and should stay a few milliseconds on an M1-class GPU.

Ink textures are finer than the previous build, so GPU memory is higher by a few tens of megabytes per display. The velocity solve stays small.

## Privacy

Wake does not use the network. There is no telemetry, no account, and no phone-home. The only thing it writes is three settings in UserDefaults: palette, intensity, and pause. Launch at login is registered with `SMAppService` and stored by macOS, not by Wake.

It does **not** request Screen Recording, Accessibility, or the camera. It never calls `CGWindowListCreateImage`, ScreenCaptureKit, or anything else that reads pixels. Window titles and window contents are ignored. Bounds and window ids are enough to stir the ink, and they are not saved.

## Build from source

Xcode 15 or later, on macOS 13 or later. The project builds for the Mac you're on (Apple Silicon or Intel). GitHub Actions builds a universal binary.

```bash
git clone https://github.com/boss-da-bot/wake.git
cd wake
./Scripts/build.sh
open build/Build/Products/Release/Wake.app
```

A local build is not quarantined, so it opens without the right-click step. To build a universal app by hand:

```bash
xcodebuild \
  -project Wake.xcodeproj \
  -scheme Wake \
  -configuration Release \
  -destination 'platform=macOS' \
  -derivedDataPath build \
  ONLY_ACTIVE_ARCH=NO \
  ARCHS="arm64 x86_64" \
  CODE_SIGN_IDENTITY="-" \
  CODE_SIGNING_REQUIRED=NO \
  build
```

There are no third-party dependencies.

## License

[MIT](LICENSE). Copyright © 2026 Wake contributors.
