#!/usr/bin/env python3
"""Generate the app icon and the README placeholder GIF. No third-party packages."""

import math
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "Wake" / "Resources" / "Assets.xcassets" / "AppIcon.appiconset"
DOCS = ROOT / "docs"


def write_png(path: Path, width: int, height: int, rgba: bytes) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = []
    stride = width * 4
    for y in range(height):
        rows.append(b"\x00" + rgba[y * stride : (y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def blend(dst, src, alpha):
    return dst + (src - dst) * alpha


def icon_pixels(size: int) -> bytes:
    pixels = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            u = x / (size - 1)
            v = y / (size - 1)
            # Soft radial field, darker at the edges.
            dx = u - 0.5
            dy = v - 0.5
            r = math.sqrt(dx * dx + dy * dy)
            shade = max(0.0, 1.0 - r * 1.35)
            bg = (
                blend(7, 22, shade),
                blend(8, 26, shade),
                blend(12, 40, shade),
            )
            # Two ink blooms.
            def blob(cx, cy, radius, color, strength):
                d = math.sqrt((u - cx) ** 2 + (v - cy) ** 2) / radius
                if d >= 1:
                    return (0, 0, 0, 0)
                falloff = (1 - d * d) ** 2
                return (*color, strength * falloff)

            blooms = [
                blob(0.42, 0.58, 0.34, (88, 112, 168), 0.85),
                blob(0.62, 0.40, 0.22, (186, 198, 214), 0.70),
                blob(0.30, 0.34, 0.16, (46, 64, 112), 0.55),
            ]
            cr, cg, cb = bg
            for br, bgc, bb, ba in blooms:
                cr = blend(cr, br, ba)
                cg = blend(cg, bgc, ba)
                cb = blend(cb, bb, ba)
            i = (y * size + x) * 4
            pixels[i : i + 4] = bytes((int(cr), int(cg), int(cb), 255))
    return bytes(pixels)


def downsample(src: bytes, src_size: int, dst_size: int) -> bytes:
    if dst_size == src_size:
        return src
    out = bytearray(dst_size * dst_size * 4)
    scale = src_size / dst_size
    for y in range(dst_size):
        for x in range(dst_size):
            x0 = int(x * scale)
            y0 = int(y * scale)
            x1 = min(src_size, int((x + 1) * scale))
            y1 = min(src_size, int((y + 1) * scale))
            if x1 <= x0:
                x1 = x0 + 1
            if y1 <= y0:
                y1 = y0 + 1
            totals = [0, 0, 0, 0]
            count = 0
            for sy in range(y0, y1):
                row = sy * src_size * 4
                for sx in range(x0, x1):
                    i = row + sx * 4
                    for channel in range(4):
                        totals[channel] += src[i + channel]
                    count += 1
            o = (y * dst_size + x) * 4
            for channel in range(4):
                out[o + channel] = totals[channel] // count
    return bytes(out)


def write_icons() -> None:
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    master = icon_pixels(1024)
    sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    for name, size in sizes.items():
        write_png(ICON_DIR / name, size, size, downsample(master, 1024, size))

    images = []
    catalog = [
        ("icon_16x16.png", "16x16", "1x"),
        ("icon_16x16@2x.png", "16x16", "2x"),
        ("icon_32x32.png", "32x32", "1x"),
        ("icon_32x32@2x.png", "32x32", "2x"),
        ("icon_128x128.png", "128x128", "1x"),
        ("icon_128x128@2x.png", "128x128", "2x"),
        ("icon_256x256.png", "256x256", "1x"),
        ("icon_256x256@2x.png", "256x256", "2x"),
        ("icon_512x512.png", "512x512", "1x"),
        ("icon_512x512@2x.png", "512x512", "2x"),
    ]
    for filename, size, scale in catalog:
        images.append(
            f'    {{\n      "filename" : "{filename}",\n      "idiom" : "mac",\n      "scale" : "{scale}",\n      "size" : "{size}"\n    }}'
        )
    contents = "{\n  \"images\" : [\n" + ",\n".join(images) + "\n  ],\n  \"info\" : {\n    \"author\" : \"xcode\",\n    \"version\" : 1\n  }\n}\n"
    (ICON_DIR / "Contents.json").write_text(contents)
    catalog_root = ICON_DIR.parent
    (catalog_root / "Contents.json").write_text('{\n  "info" : {\n    "author" : "xcode",\n    "version" : 1\n  }\n}\n')


def lzw_encode(indices: bytes, min_code_size: int) -> bytes:
    clear = 1 << min_code_size
    end = clear + 1
    code_size = min_code_size + 1
    next_code = end + 1
    table = {bytes([i]): i for i in range(clear)}
    output = bytearray()
    bit_buffer = 0
    bit_count = 0

    def write(code: int) -> None:
        nonlocal bit_buffer, bit_count, code_size
        bit_buffer |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            output.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8

    write(clear)
    prefix = b""
    for byte in indices:
        candidate = prefix + bytes([byte])
        if candidate in table:
            prefix = candidate
            continue
        write(table[prefix])
        if next_code < 4096:
            table[candidate] = next_code
            # Widen after defining the code that no longer fits in the current width,
            # so the next emitted code uses the new size. Matches GIF decoders.
            if next_code >= (1 << code_size) and code_size < 12:
                code_size += 1
            next_code += 1
        else:
            write(clear)
            table = {bytes([i]): i for i in range(clear)}
            code_size = min_code_size + 1
            next_code = end + 1
        prefix = bytes([byte])
    if prefix:
        write(table[prefix])
    write(end)
    if bit_count:
        output.append(bit_buffer & 0xFF)
    return bytes(output)


def gif_frame(indices: bytes, width: int, height: int, delay: int) -> bytes:
    min_code_size = 5  # 32-color palette
    compressed = lzw_encode(indices, min_code_size)
    blocks = bytearray()
    offset = 0
    while offset < len(compressed):
        chunk = compressed[offset : offset + 255]
        blocks.append(len(chunk))
        blocks.extend(chunk)
        offset += 255
    blocks.append(0)
    gce = struct.pack("<BBHBB", 0x04, 0x04, delay, 0, 0)
    descriptor = struct.pack("<HHHHB", 0, 0, width, height, 0)
    return b"\x21\xf9" + gce + b"\x2c" + descriptor + bytes([min_code_size]) + blocks


def palette32() -> bytes:
    colors = []
    for i in range(32):
        t = i / 31
        # Near-black through silver-blue.
        r = int(8 + t * 170)
        g = int(10 + t * 186)
        b = int(16 + t * 210)
        colors.append((min(r, 255), min(g, 255), min(b, 255)))
    raw = bytearray()
    for r, g, b in colors:
        raw.extend((r, g, b))
    return bytes(raw)


def nearest(palette, rgb) -> int:
    best = 0
    best_d = 1e9
    for index, (r, g, b) in enumerate(palette):
        d = (r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2
        if d < best_d:
            best_d = d
            best = index
    return best


def demo_gif() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    width, height = 480, 270
    frames = 16
    raw_palette = []
    blob = palette32()
    for i in range(0, len(blob), 3):
        raw_palette.append((blob[i], blob[i + 1], blob[i + 2]))

    def color_at(x, y, frame):
        u = x / width
        v = y / height
        # Moving window.
        progress = frame / (frames - 1)
        wx = 0.12 + progress * 0.62
        wy = 0.38
        ww, wh = 0.22, 0.28
        # Wake behind the window.
        ink = 0.0
        for k in range(8):
            trail = wx - 0.03 - k * 0.045
            spread = 0.02 + k * 0.012
            dx = (u - trail) / spread
            dy = (v - wy) / (0.05 + k * 0.01)
            ink += math.exp(-(dx * dx + dy * dy)) * (0.55 - k * 0.05)
        # Soft side swirls.
        for side in (-1, 1):
            sx = wx - 0.08
            sy = wy + side * 0.16
            dx = (u - sx) / 0.05
            dy = (v - sy) / 0.05
            ink += math.exp(-(dx * dx + dy * dy)) * 0.35
        ink = max(0.0, min(1.0, ink))
        # Window body stays dark, like an app sitting on the wallpaper.
        inside = abs(u - wx) < ww * 0.5 and abs(v - wy) < wh * 0.5
        if inside:
            return (18, 20, 28)
        base = (10, 12, 18)
        tint = (int(90 + ink * 140), int(110 + ink * 130), int(150 + ink * 90))
        return (
            int(base[0] + (tint[0] - base[0]) * ink),
            int(base[1] + (tint[1] - base[1]) * ink),
            int(base[2] + (tint[2] - base[2]) * ink),
        )

    # Quantize with a tiny cache so the GIF stays small.
    cache = {}
    parts = [b"GIF89a", struct.pack("<HHBBB", width, height, 0xA4, 0, 0), blob]
    parts.append(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")
    for frame in range(frames):
        indices = bytearray(width * height)
        for y in range(height):
            for x in range(width):
                rgb = color_at(x, y, frame)
                key = rgb
                index = cache.get(key)
                if index is None:
                    index = nearest(raw_palette, rgb)
                    cache[key] = index
                indices[y * width + x] = index
        parts.append(gif_frame(bytes(indices), width, height, delay=8))
    parts.append(b"\x3b")
    (DOCS / "demo.gif").write_bytes(b"".join(parts))


if __name__ == "__main__":
    write_icons()
    demo_gif()
    print("wrote icon and docs/demo.gif")
