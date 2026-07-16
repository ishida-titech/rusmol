use bytemuck::{Pod, Zeroable};

/// Per-instance data for sphere rendering
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SphereInstance {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    /// Per-instance minimum edge strength (>0 forces edge darkening even when global is 0).
    pub edge_boost: f32,
}

impl SphereInstance {
    /// Vertex buffer layout for the impostor pipeline (no mesh buffer; instance at loc 0/1/2).
    pub fn impostor_desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<SphereInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { offset: 0,     shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 4 * 3, shader_location: 1, format: wgpu::VertexFormat::Float32   },
                wgpu::VertexAttribute { offset: 4 * 4, shader_location: 2, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 4 * 7, shader_location: 3, format: wgpu::VertexFormat::Float32   },
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
    pub fn new(start: [f32; 3], end: [f32; 3], radius: f32, color: [f32; 3], edge_boost: f32) -> Self {
        Self {
            start_r:   [start[0], start[1], start[2], radius],
            end_pad:   [end[0],   end[1],   end[2],   0.0],
            color_pad: [color[0], color[1], color[2], edge_boost],
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
        // CCW winding for outward-facing triangles (so back-face culling keeps
        // the near wall, giving correct surface depth).
        indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
    }

    (vertices, indices)
}
