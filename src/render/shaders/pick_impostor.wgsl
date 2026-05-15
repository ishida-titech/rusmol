// Pick impostor shader.
// Same billboard geometry as impostor.wgsl but outputs a flat u32 instance ID
// to the R32Uint pick texture, with correct per-fragment depth from ray-sphere
// intersection so occlusion is handled properly.

struct Uniforms {
    view_proj:    mat4x4<f32>,   // offset   0
    light_dir:    vec3<f32>,     // offset  64
    _pad0:        u32,           // offset  76
    camera_pos:   vec3<f32>,     // offset  80
    _pad1:        f32,           // offset  92
    inv_proj:     mat4x4<f32>,   // offset  96
    screen_size:  vec2<f32>,     // offset 160
    surface_alpha: f32,          // offset 168
    edge_strength: f32,          // offset 172
    bg_color:      vec4<f32>,    // offset 176
    camera_right:  vec3<f32>,    // offset 192
    _pad_cr:       f32,          // offset 204
    camera_up:     vec3<f32>,    // offset 208
    _pad_cu:       f32,          // offset 220
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct InstIn {
    @location(0) inst_pos:    vec3<f32>,
    @location(1) inst_radius: f32,
    @location(2) inst_color:  vec3<f32>,
    @location(3) edge_boost:  f32,
}

struct VertOut {
    @builtin(position)              clip_pos:      vec4<f32>,
    @location(0)                    sphere_center: vec3<f32>,
    @location(1)                    sphere_radius: f32,
    @location(2)                    billboard_pos: vec3<f32>,
    @location(3) @interpolate(flat) instance_id:   u32,
}

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0)         id:    u32,
}

@vertex
fn vs_main(
    inst: InstIn,
    @builtin(vertex_index)    vid: u32,
    @builtin(instance_index)  iid: u32,
) -> VertOut {
    let offsets = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
        vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    );
    let uv = offsets[vid];

    let half_ext = inst.inst_radius * 1.15;
    let corner = inst.inst_pos
        + u.camera_right * (uv.x * half_ext)
        + u.camera_up    * (uv.y * half_ext);

    var out: VertOut;
    out.clip_pos      = u.view_proj * vec4<f32>(corner, 1.0);
    out.sphere_center = inst.inst_pos;
    out.sphere_radius = inst.inst_radius;
    out.billboard_pos = corner;
    out.instance_id   = iid + 1u;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> FragOut {
    let ray_orig = u.camera_pos;
    let ray_dir  = normalize(in.billboard_pos - u.camera_pos);

    let oc   = ray_orig - in.sphere_center;
    let b    = dot(oc, ray_dir);
    let c    = dot(oc, oc) - in.sphere_radius * in.sphere_radius;
    let disc = b * b - c;
    if disc < 0.0 { discard; }

    let sq     = sqrt(disc);
    let t_near = -b - sq;
    let t_far  = -b + sq;
    if t_far < 0.0 { discard; }

    let t   = select(t_far, t_near, t_near >= 0.0);
    let hit = ray_orig + t * ray_dir;

    let clip_hit = u.view_proj * vec4<f32>(hit, 1.0);
    let depth    = clip_hit.z / clip_hit.w;

    var out: FragOut;
    out.depth = depth;
    out.id    = in.instance_id;
    return out;
}
