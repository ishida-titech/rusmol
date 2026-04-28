use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj:         [[f32; 4]; 4], // offset   0, 64 bytes
    pub light_dir:         [f32; 3],      // offset  64, 12 bytes
    pub picked_residue_id: u32,           // offset  76,  4 bytes
    pub camera_pos:        [f32; 3],      // offset  80, 12 bytes
    pub light_intensity:   f32,           // offset  92,  4 bytes
    pub inv_proj:          [[f32; 4]; 4], // offset  96, 64 bytes (for SSAO)
    pub screen_size:       [f32; 2],      // offset 160,  8 bytes
    pub surface_alpha:     f32,           // offset 168,  4 bytes
    pub _pad:              f32,           // offset 172,  4 bytes → total 176 bytes
}

impl Uniforms {
    pub fn new(
        view_proj: Mat4,
        inv_proj: Mat4,
        light_dir: Vec3,
        camera_pos: Vec3,
        picked_residue_id: u32,
        light_intensity: f32,
        screen_size: [f32; 2],
        surface_alpha: f32,
    ) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir: light_dir.normalize().to_array(),
            picked_residue_id,
            camera_pos: camera_pos.to_array(),
            light_intensity,
            inv_proj: inv_proj.to_cols_array_2d(),
            screen_size,
            surface_alpha,
            _pad: 0.0,
        }
    }
}
