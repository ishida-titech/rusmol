// Surface shader: identical lighting to ribbon.wgsl but normals are NOT flipped
// by front-facing test, because the gradient-based normals are always outward.
struct Uniforms {
    view_proj:         mat4x4<f32>,
    light_dir:         vec3<f32>,
    picked_residue_id: u32,       // offset 76 (fills former padding after light_dir)
    // camera_pos.xyz = eye position, camera_pos.w = light_intensity
    camera_pos:        vec4<f32>,
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct VertIn {
    @location(0) position:   vec3<f32>,
    @location(1) normal:     vec3<f32>,
    @location(2) color:      vec3<f32>,
    @location(3) residue_id: u32,
}

struct VertOut {
    @builtin(position)              clip_pos:     vec4<f32>,
    @location(0)                    world_pos:    vec3<f32>,
    @location(1)                    world_normal: vec3<f32>,
    @location(2)                    color:        vec3<f32>,
    @location(3) @interpolate(flat) residue_id:   u32,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    out.clip_pos     = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos    = in.position;
    out.world_normal = in.normal;
    out.color        = in.color;
    out.residue_id   = in.residue_id;
    return out;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.0), vec3(1.0));
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    // Gradient-based normals are always outward; no front-facing flip needed.
    let N = normalize(in.world_normal);
    let L = normalize(u.light_dir);
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let H = normalize(L + V);

    // High contrast: deep shadows, bright lit face.
    let half_diff = dot(N, L) * 0.90 + 0.10;
    let ambient  = 0.10;
    let diffuse  = half_diff * 1.25;
    let specular = pow(max(dot(N, H), 0.0), 80.0) * 0.85;

    let lit = (ambient + diffuse + specular) * u.camera_pos.w;
    var color_out = in.color * lit;

    if u.picked_residue_id != 0u && in.residue_id == u.picked_residue_id {
        let rim = pow(1.0 - max(dot(V, N), 0.0), 3.0);
        color_out += rim * vec3<f32>(1.0, 0.6, 0.0) * 1.5;
    }
    return vec4<f32>(aces_tonemap(color_out), 1.0);
}
