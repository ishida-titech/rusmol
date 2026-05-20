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
    pub edge_strength:     f32,           // offset 172,  4 bytes (0=off, 1=default)
    pub bg_color:          [f32; 4],      // offset 176, 16 bytes → total 192 bytes
    pub camera_right:      [f32; 3],      // offset 192, 12 bytes (for sphere impostor billboard)
    pub roughness:         f32,           // offset 204,  4 bytes (PBR roughness, 0=mirror, 1=diffuse)
    pub camera_up:         [f32; 3],      // offset 208, 12 bytes
    pub metallic:          f32,           // offset 220,  4 bytes (PBR metallic factor)
    // ── IBL (Image-Based Lighting) ────────────────────────────────────────────
    pub sky_color:         [f32; 3],      // offset 224, 12 bytes (camera-space "sky" hemisphere)
    pub ibl_intensity:     f32,           // offset 236,  4 bytes (overall IBL scale)
    pub ground_color:      [f32; 3],      // offset 240, 12 bytes (camera-space "ground" hemisphere)
    pub shadow_strength:   f32,           // offset 252,  4 bytes (0=no shadow, 1=full shadow)
    // ── Shadow mapping ────────────────────────────────────────────────────────
    pub light_view_proj:   [[f32; 4]; 4], // offset 256, 64 bytes
    // ── Bloom ────────────────────────────────────────────────────────────────
    pub bloom_threshold:   f32,           // offset 320,  4 bytes
    pub bloom_intensity:   f32,           // offset 324,  4 bytes
    // ── Light 2 ─────────────────────────────────────────────────────────────
    pub light2_dir:        [f32; 2],      // offset 328,  8 bytes (xy; z is in next row)
    pub light2_dir_z:      f32,           // offset 336,  4 bytes
    pub light2_intensity:  f32,           // offset 340,  4 bytes
    pub _pad_end:          [f32; 2],      // offset 344,  8 bytes → total 352 bytes (aligned to 16)
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
        edge_strength: f32,
        bg_color: [f32; 3],
        camera_right: Vec3,
        camera_up: Vec3,
        roughness: f32,
        metallic: f32,
        sky_color: Vec3,
        ibl_intensity: f32,
        ground_color: Vec3,
        shadow_strength: f32,
        light_view_proj: Mat4,
        bloom_threshold: f32,
        bloom_intensity: f32,
        light2_dir: Vec3,
        light2_intensity: f32,
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
            edge_strength,
            bg_color: [bg_color[0], bg_color[1], bg_color[2], 1.0],
            camera_right: camera_right.to_array(),
            roughness,
            camera_up: camera_up.to_array(),
            metallic,
            sky_color: sky_color.to_array(),
            ibl_intensity,
            ground_color: ground_color.to_array(),
            shadow_strength,
            light_view_proj: light_view_proj.to_cols_array_2d(),
            bloom_threshold,
            bloom_intensity,
            light2_dir: [light2_dir.normalize().x, light2_dir.normalize().y],
            light2_dir_z: light2_dir.normalize().z,
            light2_intensity,
            _pad_end: [0.0; 2],
        }
    }
}

/// Uniforms for the shadow depth pass.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ShadowUniforms {
    pub light_view_proj: [[f32; 4]; 4], // offset  0, 64 bytes
    pub light_right:     [f32; 3],      // offset 64, 12 bytes
    pub _pad0:           f32,           // offset 76,  4 bytes
    pub light_up:        [f32; 3],      // offset 80, 12 bytes
    pub _pad1:           f32,           // offset 92,  4 bytes
    pub light_forward:   [f32; 3],      // offset 96, 12 bytes
    pub _pad2:           f32,           // offset108,  4 bytes → total 112 bytes
}
