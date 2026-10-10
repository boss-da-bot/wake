#include <metal_stdlib>
using namespace metal;

// 2D stable fluids. Velocity stays on a coarse grid; dye and the window
// mask are finer and sampled smoothly when drawn.
// Jos Stam, "Stable Fluids", SIGGRAPH 1999.
// Vorticity confinement: Fedkiw, Stam, Jensen, "Visual Simulation of Smoke", SIGGRAPH 2001.
//
// Velocity is stored in uv-per-second, y up, matching the windowing code.
// Dye is rgb. All simulation textures are rgba16Float; only the channels
// we write are meaningful.

struct AdvectParams {
    float dt;
    float dissipation;
    float pad0;
    float pad1;
};

struct ForceParams {
    float confinement;
    float drift;
    float time;
    float dt;
};

struct Splat {
    float4 posRadius; // xy uv, z radius in aspect-corrected uv
    float4 velocity;  // xy impulse (uv/s)
    float4 color;     // rgb ink, a amount
};

struct SplatHeader {
    uint count;
    uint mode; // 0 velocity, 1 dye
    float aspect;
    float pad;
};

struct ComposeParams {
    float4 background;
};

struct Obstacle {
    float4 rect;   // xy min uv, zw max uv
    float4 motion; // xy origin velocity (uv/s), zw size velocity (uv/s)
};

struct ObstacleHeader {
    uint count;
    uint pad0;
    float pad1;
    float pad2;
};

struct SeedParams {
    float4 color0;
    float4 color1;
    float4 color2;
    float density;
    float phase;
    float pad0;
    float pad1;
};

struct InkBloom {
    float4 posRadius;
    float4 color;
    float4 fade;
};

struct BloomHeader {
    uint count;
    float aspect;
    float pad0;
    float pad1;
};

struct VSOut {
    float4 position [[position]];
    float2 uv;
};

constexpr sampler linearClamp(coord::normalized, filter::linear, address::clamp_to_edge);
constexpr sampler nearestClamp(coord::normalized, filter::nearest, address::clamp_to_edge);

inline float4 toFloat(half4 v) {
    return float4(float(v.x), float(v.y), float(v.z), float(v.w));
}

inline half4 toHalf(float4 v) {
    return half4(half(v.x), half(v.y), half(v.z), half(v.w));
}

inline uint2 clampedCoord(int x, int y, int w, int h) {
    int cx = clamp(x, 0, w - 1);
    int cy = clamp(y, 0, h - 1);
    return uint2(uint(cx), uint(cy));
}

inline float readR(texture2d<half, access::read> tex, int x, int y, int w, int h) {
    half4 texel = tex.read(clampedCoord(x, y, w, h));
    return float(texel.x);
}

inline float2 readRG(texture2d<half, access::read> tex, int x, int y, int w, int h) {
    half4 texel = tex.read(clampedCoord(x, y, w, h));
    return float2(float(texel.x), float(texel.y));
}

/// Solid flag in x, obstacle velocity in yz. Out-of-range neighbors are fluid
/// so the screen border is not a window.
inline float4 readObstacle(texture2d<half, access::read> tex, int x, int y, int w, int h) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return float4(0.0);
    }
    return toFloat(tex.read(uint2(uint(x), uint(y))));
}

/// Obstacle mask may be finer than the velocity grid. Sample the cell center
/// in uv so a high-resolution mask still covers the right low-resolution cell.
/// Out-of-range neighbors are fluid, so the screen border is not a window.
inline float4 sampleObstacleCell(texture2d<half, access::sample> tex, int x, int y, int w, int h) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return float4(0.0);
    }
    float2 uv = (float2(x, y) + 0.5) / float2(w, h);
    return toFloat(tex.sample(nearestClamp, uv));
}

inline bool solidAt(float4 obstacle) {
    return obstacle.x > 0.5;
}

/// Separable cubic B-spline, packed into four bilinear taps.
inline float4 cubicBSpline(float v) {
    float4 n = float4(1.0, 2.0, 3.0, 4.0) - v;
    float4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return float4(x, y, z, w) * (1.0 / 6.0);
}

inline float4 sampleBicubic(texture2d<half, access::sample> tex, float2 uv) {
    float2 size = float2(tex.get_width(), tex.get_height());
    float2 inv = 1.0 / size;
    float2 coord = uv * size - 0.5;
    float2 f = fract(coord);
    coord -= f;
    float4 x = cubicBSpline(f.x);
    float4 y = cubicBSpline(f.y);
    float4 c = coord.xxyy + float2(-0.5, 1.5).xxyy;
    float4 s = float4(x.x + x.y, x.z + x.w, y.x + y.y, y.z + y.w);
    float4 safe = max(s, float4(1e-4));
    float4 offset = c + float4(x.y, x.w, y.y, y.w) / safe;
    offset *= inv.xxyy;
    float4 s0 = toFloat(tex.sample(linearClamp, offset.xz));
    float4 s1 = toFloat(tex.sample(linearClamp, offset.yz));
    float4 s2 = toFloat(tex.sample(linearClamp, offset.xw));
    float4 s3 = toFloat(tex.sample(linearClamp, offset.yw));
    float sx = safe.x / (safe.x + safe.y);
    float sy = safe.z / (safe.z + safe.w);
    return mix(mix(s3, s2, sx), mix(s1, s0, sx), sy);
}

inline float2 clampSpeed(float2 vel, float cap) {
    float speed = length(vel);
    if (speed > cap) {
        vel *= cap / speed;
    }
    return vel;
}

kernel void rasterObstacles(
    texture2d<half, access::write> dest [[texture(0)]],
    constant Obstacle *obstacles [[buffer(0)]],
    constant ObstacleHeader &header [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint w = dest.get_width();
    uint h = dest.get_height();
    if (gid.x >= w || gid.y >= h) {
        return;
    }

    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float solid = 0.0;
    float2 vel = float2(0.0);
    uint count = min(header.count, 48u);
    for (uint i = 0; i < count; ++i) {
        float4 rect = obstacles[i].rect;
        if (uv.x < rect.x || uv.y < rect.y || uv.x > rect.z || uv.y > rect.w) {
            continue;
        }
        float2 size = max(rect.zw - rect.xy, float2(1e-4));
        float2 local = saturate((uv - rect.xy) / size);
        float4 motion = obstacles[i].motion;
        vel = clampSpeed(motion.xy + motion.zw * local, 4.0);
        solid = 1.0;
    }
    dest.write(half4(half(solid), half(vel.x), half(vel.y), 0.0h), gid);
}

kernel void advect(
    texture2d<half, access::sample> source [[texture(0)]],
    texture2d<half, access::write> dest [[texture(1)]],
    texture2d<half, access::sample> velocity [[texture(2)]],
    texture2d<half, access::sample> obstacle [[texture(3)]],
    constant AdvectParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint w = dest.get_width();
    uint h = dest.get_height();
    if (gid.x >= w || gid.y >= h) {
        return;
    }

    float2 size = float2(w, h);
    float2 texel = 1.0 / size;
    float2 uv = (float2(gid) + 0.5) * texel;
    if (toFloat(obstacle.sample(nearestClamp, uv)).x > 0.5) {
        dest.write(half4(0.0h), gid);
        return;
    }
    // Dye is finer than velocity. Bicubic keeps that upsample from looking stepped.
    float2 vel = (velocity.get_width() == w)
        ? toFloat(velocity.sample(linearClamp, uv)).xy
        : sampleBicubic(velocity, uv).xy;
    float2 prevUV = clamp(uv - vel * params.dt, texel * 0.5, 1.0 - texel * 0.5);
    // Don't pull the empty interior of a window out into the fluid.
    float4 value;
    if (toFloat(obstacle.sample(nearestClamp, prevUV)).x > 0.5) {
        value = toFloat(source.sample(nearestClamp, uv)) * params.dissipation;
    } else {
        value = toFloat(source.sample(linearClamp, prevUV)) * params.dissipation;
    }
    dest.write(toHalf(value), gid);
}

kernel void splat(
    texture2d<half, access::read> source [[texture(0)]],
    texture2d<half, access::write> dest [[texture(1)]],
    texture2d<half, access::sample> obstacle [[texture(2)]],
    constant Splat *splats [[buffer(0)]],
    constant SplatHeader &header [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint w = dest.get_width();
    uint h = dest.get_height();
    if (gid.x >= w || gid.y >= h) {
        return;
    }

    if (solidAt(sampleObstacleCell(obstacle, int(gid.x), int(gid.y), int(w), int(h)))) {
        dest.write(half4(0.0h), gid);
        return;
    }

    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float4 value = toFloat(source.read(gid));
    uint count = min(header.count, 64u);

    for (uint i = 0; i < count; ++i) {
        float2 delta = uv - splats[i].posRadius.xy;
        delta.x *= header.aspect;
        float radius = max(splats[i].posRadius.z, 1e-4);
        float radius2 = radius * radius;
        float dist2 = dot(delta, delta);
        if (dist2 > radius2 * 8.0) {
            continue;
        }
        float falloff = exp(-dist2 / radius2);
        if (header.mode == 0u) {
            value.xy += splats[i].velocity.xy * falloff;
        } else {
            value.xyz += splats[i].color.xyz * splats[i].color.w * falloff;
        }
    }

    if (header.mode == 0u) {
        value.xy = clampSpeed(value.xy, 4.0);
    } else {
        value.xyz = clamp(value.xyz, float3(0.0), float3(1.35));
    }
    dest.write(toHalf(value), gid);
}

kernel void curlField(
    texture2d<half, access::read> velocity [[texture(0)]],
    texture2d<half, access::write> dest [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(dest.get_width());
    int h = int(dest.get_height());
    if (gid.x >= uint(w) || gid.y >= uint(h)) {
        return;
    }

    int x = int(gid.x);
    int y = int(gid.y);
    float left = readRG(velocity, x - 1, y, w, h).y;
    float right = readRG(velocity, x + 1, y, w, h).y;
    float bottom = readRG(velocity, x, y - 1, w, h).x;
    float top = readRG(velocity, x, y + 1, w, h).x;
    float vorticity = 0.5 * ((right - left) - (top - bottom));
    dest.write(half4(half(vorticity), 0.0h, 0.0h, 0.0h), gid);
}

kernel void confine(
    texture2d<half, access::read> velocity [[texture(0)]],
    texture2d<half, access::read> curlTex [[texture(1)]],
    texture2d<half, access::write> dest [[texture(2)]],
    texture2d<half, access::sample> obstacle [[texture(3)]],
    constant ForceParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(dest.get_width());
    int h = int(dest.get_height());
    if (gid.x >= uint(w) || gid.y >= uint(h)) {
        return;
    }

    int x = int(gid.x);
    int y = int(gid.y);
    if (solidAt(sampleObstacleCell(obstacle, x, y, w, h))) {
        dest.write(half4(0.0h), gid);
        return;
    }
    float curlL = abs(readR(curlTex, x - 1, y, w, h));
    float curlR = abs(readR(curlTex, x + 1, y, w, h));
    float curlB = abs(readR(curlTex, x, y - 1, w, h));
    float curlT = abs(readR(curlTex, x, y + 1, w, h));
    float2 gradient = float2(curlR - curlL, curlT - curlB);
    float gradientLength = length(gradient) + 1e-5;
    float2 normal = gradient / gradientLength;
    float vorticity = readR(curlTex, x, y, w, h);
    // N × ω in 2D: (Ny * ω, -Nx * ω)
    float2 spin = params.confinement * float2(normal.y, -normal.x) * vorticity;

    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float2 p = uv * 2.4;
    float n1 = sin(p.x * 1.7 + params.time * 0.11) * cos(p.y * 1.3 - params.time * 0.07);
    float n2 = cos(p.x * 1.1 - params.time * 0.09) * sin(p.y * 1.9 + params.time * 0.05);
    float2 driftForce = float2(n2, -n1) * params.drift;

    float2 vel = readRG(velocity, x, y, w, h);
    vel += (spin + driftForce) * params.dt;
    vel = clampSpeed(vel, 4.0);
    dest.write(half4(half(vel.x), half(vel.y), 0.0h, 0.0h), gid);
}

kernel void divergence(
    texture2d<half, access::read> velocity [[texture(0)]],
    texture2d<half, access::write> dest [[texture(1)]],
    texture2d<half, access::sample> obstacle [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(dest.get_width());
    int h = int(dest.get_height());
    if (gid.x >= uint(w) || gid.y >= uint(h)) {
        return;
    }

    int x = int(gid.x);
    int y = int(gid.y);
    if (solidAt(sampleObstacleCell(obstacle, x, y, w, h))) {
        dest.write(half4(0.0h), gid);
        return;
    }
    // Moving solids contribute their own velocity, so the front is a source
    // of pressure and the wake is a sink.
    float4 leftObs = sampleObstacleCell(obstacle, x - 1, y, w, h);
    float4 rightObs = sampleObstacleCell(obstacle, x + 1, y, w, h);
    float4 bottomObs = sampleObstacleCell(obstacle, x, y - 1, w, h);
    float4 topObs = sampleObstacleCell(obstacle, x, y + 1, w, h);
    float left = solidAt(leftObs) ? leftObs.y : readRG(velocity, x - 1, y, w, h).x;
    float right = solidAt(rightObs) ? rightObs.y : readRG(velocity, x + 1, y, w, h).x;
    float bottom = solidAt(bottomObs) ? bottomObs.z : readRG(velocity, x, y - 1, w, h).y;
    float top = solidAt(topObs) ? topObs.z : readRG(velocity, x, y + 1, w, h).y;
    float div = 0.5 * ((right - left) + (top - bottom));
    dest.write(half4(half(div), 0.0h, 0.0h, 0.0h), gid);
}

kernel void clearField(
    texture2d<half, access::write> dest [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= dest.get_width() || gid.y >= dest.get_height()) {
        return;
    }
    dest.write(half4(0.0h, 0.0h, 0.0h, 0.0h), gid);
}

kernel void jacobi(
    texture2d<half, access::read> pressure [[texture(0)]],
    texture2d<half, access::read> divergenceTex [[texture(1)]],
    texture2d<half, access::write> dest [[texture(2)]],
    texture2d<half, access::sample> obstacle [[texture(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(dest.get_width());
    int h = int(dest.get_height());
    if (gid.x >= uint(w) || gid.y >= uint(h)) {
        return;
    }

    int x = int(gid.x);
    int y = int(gid.y);
    if (solidAt(sampleObstacleCell(obstacle, x, y, w, h))) {
        dest.write(half4(0.0h), gid);
        return;
    }
    // Neumann: a solid neighbor copies this cell so fluid does not flow through it.
    float center = readR(pressure, x, y, w, h);
    float left = solidAt(sampleObstacleCell(obstacle, x - 1, y, w, h)) ? center : readR(pressure, x - 1, y, w, h);
    float right = solidAt(sampleObstacleCell(obstacle, x + 1, y, w, h)) ? center : readR(pressure, x + 1, y, w, h);
    float bottom = solidAt(sampleObstacleCell(obstacle, x, y - 1, w, h)) ? center : readR(pressure, x, y - 1, w, h);
    float top = solidAt(sampleObstacleCell(obstacle, x, y + 1, w, h)) ? center : readR(pressure, x, y + 1, w, h);
    float div = readR(divergenceTex, x, y, w, h);
    float solved = (left + right + bottom + top - div) * 0.25;
    dest.write(half4(half(solved), 0.0h, 0.0h, 0.0h), gid);
}

kernel void gradientSubtract(
    texture2d<half, access::read> velocity [[texture(0)]],
    texture2d<half, access::read> pressure [[texture(1)]],
    texture2d<half, access::write> dest [[texture(2)]],
    texture2d<half, access::sample> obstacle [[texture(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(dest.get_width());
    int h = int(dest.get_height());
    if (gid.x >= uint(w) || gid.y >= uint(h)) {
        return;
    }

    int x = int(gid.x);
    int y = int(gid.y);
    if (solidAt(sampleObstacleCell(obstacle, x, y, w, h))) {
        dest.write(half4(0.0h), gid);
        return;
    }
    float2 vel = float2(0.0);
    if (x > 0 && y > 0 && x + 1 < w && y + 1 < h) {
        float center = readR(pressure, x, y, w, h);
        float4 leftObs = sampleObstacleCell(obstacle, x - 1, y, w, h);
        float4 rightObs = sampleObstacleCell(obstacle, x + 1, y, w, h);
        float4 bottomObs = sampleObstacleCell(obstacle, x, y - 1, w, h);
        float4 topObs = sampleObstacleCell(obstacle, x, y + 1, w, h);
        float left = solidAt(leftObs) ? center : readR(pressure, x - 1, y, w, h);
        float right = solidAt(rightObs) ? center : readR(pressure, x + 1, y, w, h);
        float bottom = solidAt(bottomObs) ? center : readR(pressure, x, y - 1, w, h);
        float top = solidAt(topObs) ? center : readR(pressure, x, y + 1, w, h);
        vel = readRG(velocity, x, y, w, h);
        vel -= float2(right - left, top - bottom) * 0.5;
        // Fluid touching a window matches that window's normal speed.
        float countX = 0.0;
        float sumX = 0.0;
        if (solidAt(leftObs)) { sumX += leftObs.y; countX += 1.0; }
        if (solidAt(rightObs)) { sumX += rightObs.y; countX += 1.0; }
        if (countX > 0.0) { vel.x = sumX / countX; }
        float countY = 0.0;
        float sumY = 0.0;
        if (solidAt(bottomObs)) { sumY += bottomObs.z; countY += 1.0; }
        if (solidAt(topObs)) { sumY += topObs.z; countY += 1.0; }
        if (countY > 0.0) { vel.y = sumY / countY; }
        vel = clampSpeed(vel, 4.0);
    }
    dest.write(half4(half(vel.x), half(vel.y), 0.0h, 0.0h), gid);
}

kernel void seedDye(
    texture2d<half, access::write> dest [[texture(0)]],
    texture2d<half, access::read> obstacle [[texture(1)]],
    constant SeedParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint w = dest.get_width();
    uint h = dest.get_height();
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    if (solidAt(readObstacle(obstacle, int(gid.x), int(gid.y), int(w), int(h)))) {
        dest.write(half4(0.0h), gid);
        return;
    }

    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float n1 = sin(uv.x * 6.2 + params.phase) * cos(uv.y * 5.1 + 0.4);
    float n2 = sin(uv.x * 3.3 + 1.7) * sin(uv.y * 4.4 + params.phase * 0.37);
    float n3 = cos(uv.x * 8.5 - uv.y * 6.4 + 2.1);
    float field = n1 * 0.52 + n2 * 0.33 + n3 * 0.15;
    float density = saturate(0.62 + 0.28 * field) * params.density;
    float w0 = saturate(0.55 + 0.45 * n1);
    float w1 = saturate(0.55 + 0.45 * n2);
    float w2 = saturate(0.55 + 0.45 * n3);
    float weight = w0 + w1 + w2 + 1e-3;
    float3 color = (
        params.color0.xyz * w0 +
        params.color1.xyz * w1 +
        params.color2.xyz * w2
    ) / weight;
    dest.write(toHalf(float4(color * density, 0.0)), gid);
}

kernel void bloomInk(
    texture2d<half, access::read> source [[texture(0)]],
    texture2d<half, access::write> dest [[texture(1)]],
    texture2d<half, access::read> obstacle [[texture(2)]],
    constant InkBloom *blooms [[buffer(0)]],
    constant BloomHeader &header [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint w = dest.get_width();
    uint h = dest.get_height();
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    if (solidAt(readObstacle(obstacle, int(gid.x), int(gid.y), int(w), int(h)))) {
        dest.write(half4(0.0h), gid);
        return;
    }

    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float4 value = toFloat(source.read(gid));
    uint count = min(header.count, 8u);
    for (uint i = 0; i < count; ++i) {
        float2 delta = uv - blooms[i].posRadius.xy;
        delta.x *= header.aspect;
        float radius = max(blooms[i].posRadius.z, 1e-4);
        float cover = exp(-dot(delta, delta) / (radius * radius));
        value.xyz += blooms[i].color.xyz * blooms[i].color.w * cover;
        float sustain = mix(1.0, blooms[i].fade.x, cover);
        value.xyz *= sustain;
    }
    value.xyz = clamp(value.xyz, float3(0.0), float3(1.35));
    dest.write(toHalf(value), gid);
}

vertex VSOut fullscreenVertex(uint vid [[vertex_id]]) {
    // One oversized triangle. uv is 0...1 across the visible viewport.
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };
    float2 uvs[3] = {
        float2(0.0, 0.0),
        float2(2.0, 0.0),
        float2(0.0, 2.0)
    };
    VSOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

fragment float4 composeFragment(
    VSOut in [[stage_in]],
    texture2d<half> dye [[texture(0)]],
    texture2d<half> obstacle [[texture(1)]],
    constant ComposeParams &params [[buffer(0)]]
) {
    float2 uv = clamp(in.uv, 0.0, 1.0);
    float open = 1.0 - toFloat(obstacle.sample(linearClamp, uv)).x;
    // 1.2 keeps the density the old 5-tap halo produced (0.82 center + 0.38 blur).
    float3 ink = sampleBicubic(dye, uv).xyz * 1.2 * open;
    float3 mapped = ink / (1.0 + ink);
    float3 color = params.background.xyz + mapped;
    // Interleaved gradient noise, ±1/2 of an 8-bit step, breaks up banding.
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    color += (noise - 0.5) * (1.0 / 255.0);
    return float4(clamp(color, 0.0, 1.0), 1.0);
}
