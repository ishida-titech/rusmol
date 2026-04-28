// Uniforms
struct Uniforms {
    view_proj:  mat4x4<f32>,
    light_dir:  vec3<f32>,
    // camera_pos.xyz = eye position, camera_pos.w = light_intensity
    camera_pos: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

// Vertex input (icosphere mesh)
struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    // Instance data
    @location(2) inst_pos:    vec3<f32>,
    @location(3) inst_radius: f32,
    @location(4) inst_color:  vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let world_pos = in.inst_pos + in.position * in.inst_radius;
    let world_normal = in.normal; // sphere normals don't need model matrix

    var out: VertexOut;
    out.clip_pos    = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos   = world_pos;
    out.world_normal = world_normal;
    out.color       = in.inst_color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let L = normalize(u.light_dir);
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let H = normalize(L + V);

    let half_diff = dot(N, L) * 0.80 + 0.20;
    let ambient  = 0.15;
    let diffuse  = half_diff * 0.90;
    let specular = pow(max(dot(N, H), 0.0), 64.0) * 0.55;

    let lit = (ambient + diffuse + specular) * u.camera_pos.w;
    return vec4<f32>(in.color * lit, 1.0);
}
