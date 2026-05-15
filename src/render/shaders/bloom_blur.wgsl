// Separable Gaussian blur for bloom.
// Two entry points: fs_blur_h (horizontal) and fs_blur_v (vertical).
// 9-tap Gaussian (sigma ≈ 4).

@group(0) @binding(0) var src_tex:  texture_2d<f32>;
@group(0) @binding(1) var lin_samp: sampler;

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

const WEIGHTS: array<f32, 5> = array<f32, 5>(
    0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216
);

fn blur(uv: vec2<f32>, dir: vec2<f32>) -> vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(src_tex));
    let px = dir / tex_size;
    var result = textureSample(src_tex, lin_samp, uv) * WEIGHTS[0];
    for (var i = 1u; i < 5u; i++) {
        let offset = px * f32(i);
        result += textureSample(src_tex, lin_samp, uv + offset) * WEIGHTS[i];
        result += textureSample(src_tex, lin_samp, uv - offset) * WEIGHTS[i];
    }
    return result;
}

@fragment
fn fs_blur_h(in: VertOut) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_blur_v(in: VertOut) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2<f32>(0.0, 1.0));
}
