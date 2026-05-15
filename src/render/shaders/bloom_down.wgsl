// Bloom bright-pass extraction + downsample.
// Reads scene_color (Rgba16Float, full res), writes bright pixels to half-res target.

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
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var scene_tex: texture_2d<f32>;
@group(1) @binding(1) var lin_samp:  sampler;

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
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let color = textureSample(scene_tex, lin_samp, in.uv).rgb;
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let bright = max(luminance - u.bloom_threshold, 0.0);
    let contribution = bright / (luminance + 0.0001);
    return vec4<f32>(color * contribution, 1.0);
}
