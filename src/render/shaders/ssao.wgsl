// SSAO using depth reconstruction. Outputs occlusion factor in [0,1] (1=unoccluded).

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
@group(1) @binding(0) var depth_tex:  texture_depth_2d;
@group(1) @binding(1) var depth_samp: sampler;

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

// Reconstruct view-space position from depth + UV
fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc  = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let vs_h = u.inv_proj * ndc;
    return vs_h.xyz / vs_h.w;
}

fn hash21(p: vec2<f32>) -> f32 {
    let p2 = fract(p * vec2<f32>(127.1, 311.7));
    return fract((p2.x + p2.y) * (p2.x * p2.y + 43758.5453));
}

@fragment
fn fs_main(in: VertOut) -> @location(0) f32 {
    let depth = textureSample(depth_tex, depth_samp, in.uv);
    if depth >= 0.9999 { return 1.0; }

    let pixel = 1.0 / u.screen_size;
    let pos   = view_pos_from_depth(in.uv, depth);

    // Reconstruct view-space normal from neighboring pixels
    let d_r = textureSample(depth_tex, depth_samp, in.uv + vec2<f32>(pixel.x, 0.0));
    let d_u = textureSample(depth_tex, depth_samp, in.uv + vec2<f32>(0.0, pixel.y));
    let d_l = textureSample(depth_tex, depth_samp, in.uv - vec2<f32>(pixel.x, 0.0));
    let d_d = textureSample(depth_tex, depth_samp, in.uv - vec2<f32>(0.0, pixel.y));
    let p_r = view_pos_from_depth(in.uv + vec2<f32>(pixel.x, 0.0), d_r);
    let p_u = view_pos_from_depth(in.uv + vec2<f32>(0.0, pixel.y), d_u);
    let p_l = view_pos_from_depth(in.uv - vec2<f32>(pixel.x, 0.0), d_l);
    let p_d = view_pos_from_depth(in.uv - vec2<f32>(0.0, pixel.y), d_d);
    let normal = normalize(cross(p_r - p_l, p_u - p_d));

    let rot = hash21(in.uv * 3141.59) * 6.2831853;

    const N_SAMPLES: i32 = 16;
    const MAX_RADIUS_PX: f32 = 48.0;
    const RADIUS_WS: f32 = 4.0;
    const BIAS: f32 = 0.15;

    var ao = 0.0f;
    for (var i = 0i; i < N_SAMPLES; i++) {
        let fi    = f32(i);
        let phi   = fi * 2.399963 + rot;
        let r_px  = sqrt((fi + 0.5) / f32(N_SAMPLES)) * MAX_RADIUS_PX;
        let suv   = clamp(in.uv + vec2<f32>(cos(phi), sin(phi)) * r_px * pixel,
                          vec2<f32>(0.001), vec2<f32>(0.999));
        let sd    = textureSample(depth_tex, depth_samp, suv);
        let sp    = view_pos_from_depth(suv, sd);
        let dist  = length(sp - pos);
        let range = smoothstep(0.0, 1.0, RADIUS_WS / max(dist, 0.001));
        let dot_n = dot(normalize(sp - pos), normal);
        ao += max(0.0, dot_n) * range;
    }

    return clamp(1.0 - ao / f32(N_SAMPLES) * 2.0, 0.0, 1.0);
}
