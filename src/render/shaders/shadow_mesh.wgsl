// Shadow pass for ribbon / surface meshes. Depth-only (no fragment shader).

struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> su: ShadowUniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return su.light_view_proj * vec4<f32>(position, 1.0);
}
