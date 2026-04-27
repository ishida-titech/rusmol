use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj:         [[f32; 4]; 4], // offset   0, 64 bytes
    pub light_dir:         [f32; 3],      // offset  64, 12 bytes
    pub picked_residue_id: u32,           // offset  76,  4 bytes (fills former _pad0)
    pub camera_pos:        [f32; 3],      // offset  80, 12 bytes
    pub light_intensity:   f32,           // offset  92,  4 bytes → total 96 bytes
}

impl Uniforms {
    pub fn new(
        view_proj: Mat4,
        light_dir: Vec3,
        camera_pos: Vec3,
        picked_residue_id: u32,
        light_intensity: f32,
    ) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir: light_dir.normalize().to_array(),
            picked_residue_id,
            camera_pos: camera_pos.to_array(),
            light_intensity,
        }
    }
}
