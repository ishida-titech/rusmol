struct Uniforms {
    view_proj:  mat4x4<f32>,
    light_dir:  vec3<f32>,
    // camera_pos.xyz = eye position, camera_pos.w = light_intensity
    camera_pos: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    // instance
    @location(2) start_r:   vec4<f32>,  // xyz=start, w=radius
    @location(3) end_pad:   vec4<f32>,  // xyz=end,   w=unused
    @location(4) color_pad: vec4<f32>,  // xyz=color, w=unused
}

struct VertOut {
    @builtin(position) clip_pos:    vec4<f32>,
    @location(0)       world_pos:   vec3<f32>,
    @location(1)       world_normal:vec3<f32>,
    @location(2)       color:       vec3<f32>,
}

/// Rodrigues rotation: rotate `v` so that Y-axis maps to `dir`.
fn rotate_y_to(v: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let y = vec3<f32>(0.0, 1.0, 0.0);
    let c = dot(y, dir);
    if c > 0.9999 { return v; }
    if c < -0.9999 { return vec3<f32>(v.x, -v.y, v.z); }

    let ax  = normalize(cross(y, dir));
    let s   = sqrt(max(0.0, 1.0 - c * c));
    let t   = 1.0 - c;
    let ax_ = ax.x; let ay_ = ax.y; let az_ = ax.z;

    // Column-major mat3x3 (WGSL): mat3x3(col0, col1, col2)
    let rot = mat3x3<f32>(
        vec3<f32>(t*ax_*ax_ + c,       t*ax_*ay_ + s*az_, t*ax_*az_ - s*ay_),
        vec3<f32>(t*ax_*ay_ - s*az_,   t*ay_*ay_ + c,     t*ay_*az_ + s*ax_),
        vec3<f32>(t*ax_*az_ + s*ay_,   t*ay_*az_ - s*ax_, t*az_*az_ + c    )
    );
    return rot * v;
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    let start  = in.start_r.xyz;
    let radius = in.start_r.w;
    let end    = in.end_pad.xyz;
    let color  = in.color_pad.xyz;

    let axis   = end - start;
    let length = length(axis);
    let dir    = select(vec3<f32>(0.0, 1.0, 0.0), normalize(axis), length > 0.0001);
    let center = (start + end) * 0.5;

    // Scale unit cylinder: XZ by radius, Y by full length
    // (unit cylinder y ∈ [-0.5, 0.5], so scaling by length gives total height = length)
    let scaled = vec3<f32>(
        in.position.x * radius,
        in.position.y * length,
        in.position.z * radius,
    );
    let world_pos = rotate_y_to(scaled, dir) + center;

    // Side normal is purely radial (XZ direction of mesh vertex)
    let mesh_normal = normalize(vec3<f32>(in.normal.x, 0.0, in.normal.z));
    let world_normal = rotate_y_to(mesh_normal, dir);

    var out: VertOut;
    out.clip_pos    = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos   = world_pos;
    out.world_normal = world_normal;
    out.color       = color;
    return out;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.0), vec3(1.0));
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let L = normalize(u.light_dir);
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let H = normalize(L + V);

    let half_diff = dot(N, L) * 0.80 + 0.20;
    let ambient  = 0.15;
    let diffuse  = half_diff * 0.90;
    let specular = pow(max(dot(N, H), 0.0), 64.0) * 0.55;

    let lit = (ambient + diffuse + specular) * u.camera_pos.w;
    return vec4<f32>(aces_tonemap(in.color * lit), 1.0);
}
