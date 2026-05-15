// Shadow pass for sphere impostors.
// Orthographic billboard from light's perspective with ray-sphere depth.

struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,   // offset   0
    light_right:     vec3<f32>,     // offset  64
    _pad0:           f32,           // offset  76
    light_up:        vec3<f32>,     // offset  80
    _pad1:           f32,           // offset  92
    light_forward:   vec3<f32>,     // offset  96
    _pad2:           f32,           // offset 108
}

@group(0) @binding(0) var<uniform> su: ShadowUniforms;

struct InstIn {
    @location(0) inst_pos:    vec3<f32>,
    @location(1) inst_radius: f32,
}

struct VertOut {
    @builtin(position) clip_pos:      vec4<f32>,
    @location(0)       sphere_center: vec3<f32>,
    @location(1)       sphere_radius: f32,
    @location(2)       billboard_pos: vec3<f32>,
}

@vertex
fn vs_main(inst: InstIn, @builtin(vertex_index) vid: u32) -> VertOut {
    let offsets = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
        vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    );
    let uv = offsets[vid];
    let half_ext = inst.inst_radius * 1.15;
    let corner = inst.inst_pos
        + su.light_right * (uv.x * half_ext)
        + su.light_up    * (uv.y * half_ext);

    var out: VertOut;
    out.clip_pos      = su.light_view_proj * vec4<f32>(corner, 1.0);
    out.sphere_center = inst.inst_pos;
    out.sphere_radius = inst.inst_radius;
    out.billboard_pos = corner;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @builtin(frag_depth) f32 {
    let ray_orig = in.billboard_pos;
    let ray_dir  = su.light_forward;

    let oc   = ray_orig - in.sphere_center;
    let b    = dot(oc, ray_dir);
    let c    = dot(oc, oc) - in.sphere_radius * in.sphere_radius;
    let disc = b * b - c;
    if disc < 0.0 { discard; }

    let t   = -b - sqrt(disc);
    let hit = ray_orig + t * ray_dir;

    let clip = su.light_view_proj * vec4<f32>(hit, 1.0);
    return clip.z / clip.w;
}
