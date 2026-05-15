// Shadow pass for cylinders. Depth-only (no fragment shader needed).

struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> su: ShadowUniforms;

struct VertIn {
    @location(0) position: vec3<f32>,
    // location 1 (normal) skipped — not needed for shadow
    @location(2) start_r:  vec4<f32>,
    @location(3) end_pad:  vec4<f32>,
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
    let rot = mat3x3<f32>(
        vec3<f32>(t*ax_*ax_ + c,       t*ax_*ay_ + s*az_, t*ax_*az_ - s*ay_),
        vec3<f32>(t*ax_*ay_ - s*az_,   t*ay_*ay_ + c,     t*ay_*az_ + s*ax_),
        vec3<f32>(t*ax_*az_ + s*ay_,   t*ay_*az_ - s*ax_, t*az_*az_ + c    )
    );
    return rot * v;
}

@vertex
fn vs_main(in: VertIn) -> @builtin(position) vec4<f32> {
    let start  = in.start_r.xyz;
    let radius = in.start_r.w;
    let end    = in.end_pad.xyz;
    let axis   = end - start;
    let len    = length(axis);
    let dir    = select(vec3<f32>(0.0, 1.0, 0.0), normalize(axis), len > 0.0001);
    let center = (start + end) * 0.5;
    let scaled = vec3<f32>(in.position.x * radius, in.position.y * len, in.position.z * radius);
    let world_pos = rotate_y_to(scaled, dir) + center;
    return su.light_view_proj * vec4<f32>(world_pos, 1.0);
}
