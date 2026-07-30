// Depth-of-field pass. Reads the resolved HDR scene color (Rgba16Float, linear)
// plus single-sample depth, computes a per-pixel circle of confusion from the
// view-space distance to the focus plane, and gathers a disk blur scaled by it.
// A no-op (returns the original sample) when dof_strength == 0, so it is free by
// default. Output is HDR linear color, copied back over scene_color downstream.

struct Uniforms {
    view_proj:         mat4x4<f32>,
    light_dir:         vec3<f32>,
    picked_residue_id: u32,
    camera_pos:        vec3<f32>,
    light_intensity:   f32,
    inv_proj:          mat4x4<f32>,
    screen_size:       vec2<f32>,
    surface_alpha:     f32,
    edge_strength:     f32,
    bg_color:          vec4<f32>,
    camera_right:      vec3<f32>,
    roughness:         f32,
    camera_up:         vec3<f32>,
    metallic:          f32,
    sky_color:         vec3<f32>,
    ibl_intensity:     f32,
    ground_color:      vec3<f32>,
    shadow_strength:   f32,
    light_view_proj:   mat4x4<f32>,
    bloom_threshold:   f32,
    bloom_intensity:   f32,
    light2_dir:        vec2<f32>,
    light2_dir_z:      f32,
    light2_intensity:  f32,
    bg_transparent:    u32,
    ssao_samples:      u32,
    dof_strength:      f32,          // offset 352
    dof_focus:         f32,          // offset 356
    dof_aperture:      f32,          // offset 360
    _pad_end:          f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var scene_tex: texture_2d<f32>;   // resolved scene (Rgba16Float, linear HDR)
@group(1) @binding(1) var depth_tex: texture_depth_2d;  // resolved depth
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

// View-space Z (negative, camera looks down -Z) reconstructed from depth.
fn view_z_from_depth(uv: vec2<f32>, depth: f32) -> f32 {
    let ndc  = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let vs_h = u.inv_proj * ndc;
    return (vs_h.xyz / vs_h.w).z;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let base = textureSample(scene_tex, samp, in.uv);
    if u.dof_strength <= 0.0 { return base; }

    let depth = textureSample(depth_tex, samp, in.uv);
    // Keep the empty background sharp/clean (avoids halos bleeding outward).
    if depth >= 0.9999 { return base; }

    let dist = -view_z_from_depth(in.uv, depth);   // distance from camera, world Å
    const MAX_COC_PX: f32 = 24.0;
    let coc = clamp(abs(dist - u.dof_focus) * u.dof_aperture * u.dof_strength,
                    0.0, MAX_COC_PX);
    if coc < 0.5 { return base; }

    let pixel = 1.0 / u.screen_size;
    // Golden-angle disk gather (16 taps + center).
    const N: i32 = 16;
    var acc = base;
    var wsum = 1.0;
    for (var i = 0i; i < N; i++) {
        let fi  = f32(i);
        let phi = fi * 2.399963;
        let r   = sqrt((fi + 0.5) / f32(N)) * coc;
        let suv = in.uv + vec2<f32>(cos(phi), sin(phi)) * r * pixel;
        acc += textureSample(scene_tex, samp, suv);
        wsum += 1.0;
    }
    return acc / wsum;
}
