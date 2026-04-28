// Resolves 4× MSAA depth to single-sample depth texture.
// Drawn as a full-screen triangle (3 vertices, no vertex buffer).

@group(0) @binding(0) var ms_depth: texture_depth_multisampled_2d;

struct VertOut { @builtin(position) pos: vec4<f32> }

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertOut {
    let x = f32(vi & 1u) * 4.0 - 1.0;
    let y = f32((vi >> 1u) & 1u) * 4.0 - 1.0;
    return VertOut(vec4<f32>(x, y, 0.0, 1.0));
}

@fragment
fn fs_main(in: VertOut) -> @builtin(frag_depth) f32 {
    let c = vec2<i32>(i32(in.pos.x), i32(in.pos.y));
    var d = 1.0f;
    for (var s = 0i; s < 4i; s++) {
        d = min(d, textureLoad(ms_depth, c, s));
    }
    return d;
}
