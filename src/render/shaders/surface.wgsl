// Surface shader: Weighted Blended OIT output.
// Outputs accum (Rgba16Float) at @location(0) and reveal (R16Float) at @location(1).
// ACES tone mapping is done in post.wgsl; this shader outputs linear HDR.

struct Uniforms {
    view_proj:         mat4x4<f32>,  // offset 0
    light_dir:         vec3<f32>,    // offset 64
    picked_residue_id: u32,          // offset 76
    camera_pos:        vec3<f32>,    // offset 80
    light_intensity:   f32,          // offset 92
    inv_proj:          mat4x4<f32>,  // offset 96
    screen_size:       vec2<f32>,    // offset 160
    surface_alpha:     f32,          // offset 168
    _pad:              f32,          // offset 172
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertIn {
    @location(0) position:   vec3<f32>,
    @location(1) normal:     vec3<f32>,
    @location(2) color:      vec3<f32>,
    @location(3) residue_id: u32,
}

struct VertOut {
    @builtin(position)              clip_pos:   vec4<f32>,
    @location(0)                    world_pos:  vec3<f32>,
    @location(1)                    world_nrm:  vec3<f32>,
    @location(2)                    color:      vec3<f32>,
    @location(3) @interpolate(flat) residue_id: u32,
}

struct OITOut {
    @location(0) accum:  vec4<f32>,  // RGB*alpha*w, alpha*w
    @location(1) reveal: vec4<f32>,  // .a = alpha (used by blend equation)
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    out.clip_pos   = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos  = in.position;
    out.world_nrm  = in.normal;
    out.color      = in.color;
    out.residue_id = in.residue_id;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> OITOut {
    let N = normalize(in.world_nrm);
    let L = normalize(u.light_dir);
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let H = normalize(L + V);

    let half_diff = dot(N, L) * 0.90 + 0.10;
    let ambient  = 0.10;
    let diffuse  = half_diff * 1.25;
    let specular = pow(max(dot(N, H), 0.0), 80.0) * 0.85;
    let lit = (ambient + diffuse + specular) * u.light_intensity;
    var color = in.color * lit;

    if u.picked_residue_id != 0u && in.residue_id == u.picked_residue_id {
        let rim = pow(1.0 - max(dot(V, N), 0.0), 3.0);
        color += rim * vec3<f32>(1.0, 0.6, 0.0) * 1.5;
    }

    let alpha = clamp(u.surface_alpha, 0.0, 1.0);

    // WB-OIT weight: depth-dependent
    let depth = in.clip_pos.z / in.clip_pos.w;
    let w = alpha * clamp(0.03 / (1e-5 + pow(depth * 0.2, 3.0)), 1e-2, 3e3);

    var out: OITOut;
    out.accum  = vec4<f32>(color * alpha * w, alpha * w);
    out.reveal = vec4<f32>(0.0, 0.0, 0.0, alpha);  // .a used as src_alpha for blend
    return out;
}
