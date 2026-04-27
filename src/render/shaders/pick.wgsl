// Pick shader: renders sphere instances with flat instance_index+1 color to a R32Uint target.
// The fragment output is a u32 ID (0 = background, N = sphere instance N).
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct VertIn {
    @location(0) position:    vec3<f32>,
    @location(1) normal:      vec3<f32>,
    @location(2) inst_pos:    vec3<f32>,
    @location(3) inst_radius: f32,
    @location(4) inst_color:  vec3<f32>,
}

struct VertOut {
    @builtin(position)                 clip_pos:    vec4<f32>,
    @location(0) @interpolate(flat)    instance_id: u32,
}

@vertex
fn vs_main(in: VertIn, @builtin(instance_index) inst_idx: u32) -> VertOut {
    let world_pos = in.inst_pos + in.position * in.inst_radius;
    var out: VertOut;
    out.clip_pos    = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.instance_id = inst_idx + 1u;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) u32 {
    return in.instance_id;
}
