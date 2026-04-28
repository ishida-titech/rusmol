// Post-process composite pass:
//   1. Blend WB-OIT transparent surface onto opaque scene
//   2. Apply SSAO
//   3. Apply depth Sobel edge outline
//   4. Apply ACES tone mapping

struct Uniforms {
    view_proj:         mat4x4<f32>,
    light_dir:         vec3<f32>,
    picked_residue_id: u32,
    camera_pos:        vec3<f32>,
    light_intensity:   f32,
    inv_proj:          mat4x4<f32>,
    screen_size:       vec2<f32>,
    _pad:              vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var scene_tex:  texture_2d<f32>;   // opaque scene (Rgba16Float)
@group(1) @binding(1) var accum_tex:  texture_2d<f32>;   // OIT accum (Rgba16Float)
@group(1) @binding(2) var reveal_tex: texture_2d<f32>;   // OIT reveal (R16Float)
@group(1) @binding(3) var ssao_tex:   texture_2d<f32>;   // SSAO (R8Unorm)
@group(1) @binding(4) var depth_tex:  texture_depth_2d;  // resolved depth
@group(1) @binding(5) var lin_samp:   sampler;

struct VertOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertOut {
    let x = f32(vi & 1u) * 4.0 - 1.0;
    let y = f32((vi >> 1u) & 1u) * 4.0 - 1.0;
    var out: VertOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv  = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.0), vec3(1.0));
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let scene  = textureSample(scene_tex,  lin_samp, in.uv).rgb;
    let accum  = textureSample(accum_tex,  lin_samp, in.uv);
    let reveal = textureSample(reveal_tex, lin_samp, in.uv).r;
    let ao     = textureSample(ssao_tex,   lin_samp, in.uv).r;

    // WB-OIT composite
    var color = scene;
    if accum.a > 1e-4 {
        let avg = accum.rgb / max(accum.a, 1e-5);
        color = avg * (1.0 - reveal) + color * reveal;
    }

    // SSAO: darken ambient term
    let ao_strength = 0.7;
    color *= mix(1.0, ao, ao_strength);

    // Depth-based edge outline (Sobel)
    let px = 1.0 / u.screen_size;
    let d00 = textureSample(depth_tex, lin_samp, in.uv + vec2<f32>(-px.x, -px.y));
    let d10 = textureSample(depth_tex, lin_samp, in.uv + vec2<f32>( 0.0,  -px.y));
    let d20 = textureSample(depth_tex, lin_samp, in.uv + vec2<f32>( px.x, -px.y));
    let d01 = textureSample(depth_tex, lin_samp, in.uv + vec2<f32>(-px.x,  0.0 ));
    let d21 = textureSample(depth_tex, lin_samp, in.uv + vec2<f32>( px.x,  0.0 ));
    let d02 = textureSample(depth_tex, lin_samp, in.uv + vec2<f32>(-px.x,  px.y));
    let d12 = textureSample(depth_tex, lin_samp, in.uv + vec2<f32>( 0.0,   px.y));
    let d22 = textureSample(depth_tex, lin_samp, in.uv + vec2<f32>( px.x,  px.y));

    let gx = (-d00 - 2.0*d01 - d02) + (d20 + 2.0*d21 + d22);
    let gy = (-d00 - 2.0*d10 - d20) + (d02 + 2.0*d12 + d22);
    let edge = sqrt(gx*gx + gy*gy);
    let outline = smoothstep(0.003, 0.015, edge) * 0.85;
    color = color * (1.0 - outline);

    return vec4<f32>(aces_tonemap(color), 1.0);
}
