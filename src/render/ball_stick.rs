use bytemuck::{Pod, Zeroable};

/// Per-instance data for sphere rendering
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SphereInstance {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub _pad: f32,
}

impl SphereInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<SphereInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // position (location 2)
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // radius (location 3)
                wgpu::VertexAttribute {
                    offset: 4 * 3,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                // color (location 4)
                wgpu::VertexAttribute {
                    offset: 4 * 4,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// A vertex in the icosphere mesh
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 4 * 3,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Generate a unit icosphere with `subdivisions` levels.
/// Returns (vertices, indices).
pub fn icosphere(subdivisions: u32) -> (Vec<Vertex>, Vec<u32>) {
    // Start from icosahedron
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let mut positions: Vec<[f32; 3]> = vec![
        norm([-1.0,  t,  0.0]), norm([ 1.0,  t,  0.0]),
        norm([-1.0, -t,  0.0]), norm([ 1.0, -t,  0.0]),
        norm([ 0.0, -1.0,  t]), norm([ 0.0,  1.0,  t]),
        norm([ 0.0, -1.0, -t]), norm([ 0.0,  1.0, -t]),
        norm([ t,  0.0, -1.0]), norm([ t,  0.0,  1.0]),
        norm([-t,  0.0, -1.0]), norm([-t,  0.0,  1.0]),
    ];
    let mut faces: Vec<[u32; 3]> = vec![
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ];

    use std::collections::HashMap;
    let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();

    let mut get_midpoint = |positions: &mut Vec<[f32; 3]>, a: u32, b: u32| -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&idx) = midpoint_cache.get(&key) {
            return idx;
        }
        let pa = positions[a as usize];
        let pb = positions[b as usize];
        let mid = norm([
            (pa[0] + pb[0]) * 0.5,
            (pa[1] + pb[1]) * 0.5,
            (pa[2] + pb[2]) * 0.5,
        ]);
        let idx = positions.len() as u32;
        positions.push(mid);
        midpoint_cache.insert(key, idx);
        idx
    };

    for _ in 0..subdivisions {
        let mut new_faces = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in &faces {
            let ab = get_midpoint(&mut positions, *a, *b);
            let bc = get_midpoint(&mut positions, *b, *c);
            let ca = get_midpoint(&mut positions, *c, *a);
            new_faces.push([*a, ab, ca]);
            new_faces.push([*b, bc, ab]);
            new_faces.push([*c, ca, bc]);
            new_faces.push([ab, bc, ca]);
        }
        faces = new_faces;
    }

    let vertices: Vec<Vertex> = positions.iter().map(|&p| Vertex { position: p, normal: p }).collect();
    let indices: Vec<u32> = faces.iter().flat_map(|f| f.iter().copied()).collect();
    (vertices, indices)
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    [v[0]/len, v[1]/len, v[2]/len]
}

// ── Cylinder ──────────────────────────────────────────────────────────────────

/// Per-instance data for cylinder (half-bond) rendering.
/// Packed as three vec4 for clean WGSL attribute binding.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CylinderInstance {
    /// start.xyz + radius as w
    pub start_r: [f32; 4],
    /// end.xyz + padding
    pub end_pad: [f32; 4],
    /// color.rgb + padding
    pub color_pad: [f32; 4],
}

impl CylinderInstance {
    pub fn new(start: [f32; 3], end: [f32; 3], radius: f32, color: [f32; 3]) -> Self {
        Self {
            start_r:   [start[0], start[1], start[2], radius],
            end_pad:   [end[0],   end[1],   end[2],   0.0],
            color_pad: [color[0], color[1], color[2], 0.0],
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<CylinderInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { offset: 0,  shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 16, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 32, shader_location: 4, format: wgpu::VertexFormat::Float32x4 },
            ],
        }
    }
}

/// Generate a unit cylinder along Y axis (y ∈ [-0.5, 0.5], radius = 1).
/// No end caps (they are hidden inside atom spheres).
pub fn gen_cylinder(segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let n = segments as usize;
    let mut vertices = Vec::with_capacity(n * 2);
    let mut indices  = Vec::with_capacity(n * 6);

    for i in 0..n {
        let angle = i as f32 * 2.0 * std::f32::consts::PI / n as f32;
        let (s, c) = angle.sin_cos();
        let normal = [c, 0.0, s];
        vertices.push(Vertex { position: [c, -0.5, s], normal });
        vertices.push(Vertex { position: [c,  0.5, s], normal });
    }

    for i in 0..n as u32 {
        let i0 = i * 2;
        let i1 = i * 2 + 1;
        let i2 = ((i + 1) % n as u32) * 2;
        let i3 = ((i + 1) % n as u32) * 2 + 1;
        // two triangles per quad
        indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
    }

    (vertices, indices)
}
