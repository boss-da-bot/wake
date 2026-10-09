#include <metal_stdlib>
using namespace metal;

// 2D stable fluids on a low-resolution grid.
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

struct VSOut {
    float4 position [[position]];
    float2 uv;
};

constexpr sampler linearClamp(coord::normalized, filter::linear, address::clamp_to_edge);

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

inline float2 clampSpeed(float2 vel, float cap) {
    float speed = length(vel);
    if (speed > cap) {
        vel *= cap / speed;
    }
    return vel;
}

kernel void advect(
    texture2d<half, access::sample> source [[texture(0)]],
    texture2d<half, access::write> dest [[texture(1)]],
    texture2d<half, access::sample> velocity [[texture(2)]],
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
    float2 vel = toFloat(velocity.sample(linearClamp, uv)).xy;
    float2 prevUV = clamp(uv - vel * params.dt, texel * 0.5, 1.0 - texel * 0.5);
    float4 value = toFloat(source.sample(linearClamp, prevUV)) * params.dissipation;
    dest.write(toHalf(value), gid);
}

kernel void splat(
    texture2d<half, access::read> source [[texture(0)]],
    texture2d<half, access::write> dest [[texture(1)]],
    constant Splat *splats [[buffer(0)]],
    constant SplatHeader &header [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint w = dest.get_width();
    uint h = dest.get_height();
    if (gid.x >= w || gid.y >= h) {
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
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(dest.get_width());
    int h = int(dest.get_height());
    if (gid.x >= uint(w) || gid.y >= uint(h)) {
        return;
    }

    int x = int(gid.x);
    int y = int(gid.y);
    float left = readRG(velocity, x - 1, y, w, h).x;
    float right = readRG(velocity, x + 1, y, w, h).x;
    float bottom = readRG(velocity, x, y - 1, w, h).y;
    float top = readRG(velocity, x, y + 1, w, h).y;
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
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(dest.get_width());
    int h = int(dest.get_height());
    if (gid.x >= uint(w) || gid.y >= uint(h)) {
        return;
    }

    int x = int(gid.x);
    int y = int(gid.y);
    float left = readR(pressure, x - 1, y, w, h);
    float right = readR(pressure, x + 1, y, w, h);
    float bottom = readR(pressure, x, y - 1, w, h);
    float top = readR(pressure, x, y + 1, w, h);
    float div = readR(divergenceTex, x, y, w, h);
    float solved = (left + right + bottom + top - div) * 0.25;
    dest.write(half4(half(solved), 0.0h, 0.0h, 0.0h), gid);
}

kernel void gradientSubtract(
    texture2d<half, access::read> velocity [[texture(0)]],
    texture2d<half, access::read> pressure [[texture(1)]],
    texture2d<half, access::write> dest [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(dest.get_width());
    int h = int(dest.get_height());
    if (gid.x >= uint(w) || gid.y >= uint(h)) {
        return;
    }

    int x = int(gid.x);
    int y = int(gid.y);
    float2 vel = float2(0.0);
    if (x > 0 && y > 0 && x + 1 < w && y + 1 < h) {
        float left = readR(pressure, x - 1, y, w, h);
        float right = readR(pressure, x + 1, y, w, h);
        float bottom = readR(pressure, x, y - 1, w, h);
        float top = readR(pressure, x, y + 1, w, h);
        vel = readRG(velocity, x, y, w, h);
        vel -= float2(right - left, top - bottom) * 0.5;
        vel = clampSpeed(vel, 4.0);
    }
    dest.write(half4(half(vel.x), half(vel.y), 0.0h, 0.0h), gid);
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
    constant ComposeParams &params [[buffer(0)]]
) {
    float2 uv = clamp(in.uv, 0.0, 1.0);
    float2 texel = 1.5 / float2(dye.get_width(), dye.get_height());
    float3 center = toFloat(dye.sample(linearClamp, uv)).xyz;
    float3 halo = (
        toFloat(dye.sample(linearClamp, uv + float2(texel.x, 0.0))).xyz +
        toFloat(dye.sample(linearClamp, uv - float2(texel.x, 0.0))).xyz +
        toFloat(dye.sample(linearClamp, uv + float2(0.0, texel.y))).xyz +
        toFloat(dye.sample(linearClamp, uv - float2(0.0, texel.y))).xyz
    ) * 0.25;
    float3 ink = center * 0.82 + halo * 0.38;
    float3 mapped = ink / (1.0 + ink);
    float3 color = clamp(params.background.xyz + mapped, 0.0, 1.0);
    return float4(color, 1.0);
}
