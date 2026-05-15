// SSAO blur: depth-aware 5×5 bilateral box filter.
// Preserves occlusion edges by down-weighting samples whose depth
// differs significantly from the center pixel.

struct Uniforms {
    view_proj:    mat4x4<f32>,
    light_dir:    vec3<f32>,
    _pad0:        u32,
    camera_pos:   vec3<f32>,
    _pad1:        f32,
    inv_proj:     mat4x4<f32>,
    screen_size:  vec2<f32>,
    // remaining fields unused here
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var ssao_tex:  texture_2d<f32>;
@group(1) @binding(1) var depth_tex: texture_depth_2d;
@group(1) @binding(2) var samp:      sampler;

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

@fragment
fn fs_main(in: VertOut) -> @location(0) f32 {
    let px = 1.0 / u.screen_size;

    let center_d = textureSample(depth_tex, samp, in.uv);
    // Background: no occlusion
    if center_d >= 0.9999 { return 1.0; }

    var ao_sum = 0.0;
    var w_sum  = 0.0;

    // 5×5 box (radius 2 pixels)
    for (var i = -2; i <= 2; i++) {
        for (var j = -2; j <= 2; j++) {
            let off  = vec2<f32>(f32(i), f32(j)) * px;
            let s_uv = clamp(in.uv + off, vec2<f32>(0.001), vec2<f32>(0.999));
            let s_d  = textureSample(depth_tex, samp, s_uv);
            // Depth-based weight: sharply reduces when depth differs (edge preservation)
            let w    = exp(-abs(s_d - center_d) * 2000.0);
            ao_sum  += textureSample(ssao_tex, samp, s_uv).r * w;
            w_sum   += w;
        }
    }

    return ao_sum / w_sum;
}
