use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use std::collections::HashMap;

use crate::render::ball_stick::{CylinderInstance, SphereInstance, Vertex};
use crate::render::camera::Camera;
use crate::render::picker::Picker;
use crate::render::ribbon::{build_ribbon, residues_consecutive, RibbonGap, RibbonVertex};
use crate::render::uniform::ShadowUniforms;
use crate::render::surface::build_surface;
use crate::structure::rings::detect_aromatic_rings;
use crate::render::uniform::Uniforms;
use crate::scene::object::{RepresentationType, REP_BACKBONE, REP_BALL_STICK, REP_LINES, REP_RIBBON, REP_STICK, REP_SURFACE};
use crate::scene::{AtomRef, Scene};
use crate::util::color::vdw_radius;

/// Result of a pick operation: either a direct atom hit (BallAndStick) or a
/// residue-level hit found via ghost-sphere nearest search (Ribbon / Surface).
pub enum PickResult {
    /// Direct hit on a rendered sphere — show atom-level info.
    Atom(crate::scene::AtomRef),
    /// Nearest ghost-sphere hit — show residue-level info only.
    Residue(crate::scene::AtomRef),
}

const BOND_RADIUS: f32 = 0.18;
/// Stick representation is slightly chunkier than ball-and-stick bonds.
const STICK_RADIUS: f32 = BOND_RADIUS * 2.5;
const BACKBONE_TUBE_RADIUS: f32 = 0.30;
const BACKBONE_JOINT_RADIUS: f32 = 0.36;
const SHADOW_MAP_SIZE: u32 = 2048;

const AROMATIC_RING_RADIUS: f32 = 0.04;
const AROMATIC_RING_SEGMENTS: usize = 24;
const AROMATIC_RING_SCALE: f32 = 0.58;

/// Pocket-surface clip: a surface triangle is kept when the average of its
/// vertices' `outward_normal · dir_to_nearest_ligand` exceeds this threshold.
/// Negative so the pocket wall and its rim are kept generously; the far side
/// (normals pointing away) is dropped, then isolated fragments are pruned.
const SURFACE_POCKET_FACING: f32 = -0.35;

/// After the pocket clip, drop connected mesh components whose triangle count
/// is below this fraction of the largest component (removes stray back-side bits).
const SURFACE_POCKET_MIN_COMPONENT: f32 = 0.2;

/// Boundary loops with at most this many edges are treated as holes and filled;
/// the single largest loop (the intended open rim) is always left open.
const SURFACE_HOLE_MAX_EDGES: usize = 80;

const DASH_RADIUS: f32 = 0.08;
const DASH_LEN: f32 = 0.6;
const GAP_LEN: f32 = 0.4;

/// Emit dashed cylinders between two points (for missing-residue gaps).
fn emit_dashed_cylinders(
    cylinders: &mut Vec<CylinderInstance>,
    p1: &[f32; 3],
    p2: &[f32; 3],
    color1: &[f32; 3],
    color2: &[f32; 3],
) {
    let dx = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let total = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
    if total < 1e-4 { return; }
    let dir = [dx[0] / total, dx[1] / total, dx[2] / total];
    let stride = DASH_LEN + GAP_LEN;
    let mut t = 0.0f32;
    while t < total {
        let t_end = (t + DASH_LEN).min(total);
        let a = [p1[0] + dir[0] * t, p1[1] + dir[1] * t, p1[2] + dir[2] * t];
        let b = [p1[0] + dir[0] * t_end, p1[1] + dir[1] * t_end, p1[2] + dir[2] * t_end];
        let frac = (t + t_end) * 0.5 / total;
        let col = if frac < 0.5 { *color1 } else { *color2 };
        cylinders.push(CylinderInstance::new(a, b, DASH_RADIUS, col, 0.0));
        t += stride;
    }
}

pub struct RenderState {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    uniform_bind_group_layout: wgpu::BindGroupLayout,

    // ── Sphere impostor pipeline ──────────────────────────────────────────────
    sphere_pipeline: wgpu::RenderPipeline,
    sphere_instances: Option<wgpu::Buffer>,
    sphere_instance_count: u32,

    // ── Ligand overlay (drawn after surface so ligands are always opaque) ────
    ligand_sphere_pipeline: wgpu::RenderPipeline,
    ligand_cylinder_pipeline: wgpu::RenderPipeline,
    ligand_sphere_instances: Option<wgpu::Buffer>,
    ligand_sphere_instance_count: u32,
    ligand_cylinder_instances: Option<wgpu::Buffer>,
    ligand_cylinder_instance_count: u32,

    // ── Cylinder pipeline ────────────────────────────────────────────────────
    cylinder_pipeline: wgpu::RenderPipeline,
    cylinder_vb: wgpu::Buffer,
    cylinder_ib: wgpu::Buffer,
    cylinder_index_count: u32,
    cylinder_instances: Option<wgpu::Buffer>,
    cylinder_instance_count: u32,

    // ── Ribbon pipeline ───────────────────────────────────────────────────────
    ribbon_pipeline: wgpu::RenderPipeline,
    ribbon_vb: Option<wgpu::Buffer>,
    ribbon_ib: Option<wgpu::Buffer>,
    ribbon_index_count: u32,

    // ── Surface pipeline (alpha-blend into MSAA target) ──────────────────────
    surface_pipeline: wgpu::RenderPipeline,
    surface_vb: Option<wgpu::Buffer>,
    surface_ib: Option<wgpu::Buffer>,
    surface_index_count: u32,

    // ── MSAA depth (multisampled) ─────────────────────────────────────────────
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,

    // ── MSAA × 4 color ───────────────────────────────────────────────────────
    msaa_texture: wgpu::Texture,
    msaa_color_view: wgpu::TextureView,

    // ── Opaque scene resolve target (Rgba16Float, sample_count=1) ─────────────
    scene_color_tex: wgpu::Texture,
    scene_color_view: wgpu::TextureView,

    // ── Single-sample depth for post-process sampling ─────────────────────────
    depth_single_tex: wgpu::Texture,
    depth_single_view: wgpu::TextureView,

    // ── SSAO texture ──────────────────────────────────────────────────────────
    ssao_tex: wgpu::Texture,
    ssao_view: wgpu::TextureView,

    // ── Post-process pipelines ────────────────────────────────────────────────
    depth_resolve_pipeline: wgpu::RenderPipeline,
    ssao_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,

    // ── Depth resolve bind group ──────────────────────────────────────────────
    depth_resolve_bgl: wgpu::BindGroupLayout,
    depth_resolve_bg: wgpu::BindGroup,

    // ── SSAO bind group ───────────────────────────────────────────────────────
    ssao_bgl: wgpu::BindGroupLayout,
    ssao_bg: wgpu::BindGroup,

    // ── SSAO blur (depth-aware 5×5 bilateral) ────────────────────────────────
    ssao_blur_tex: wgpu::Texture,
    ssao_blur_view: wgpu::TextureView,
    ssao_blur_pipeline: wgpu::RenderPipeline,
    ssao_blur_bgl: wgpu::BindGroupLayout,
    ssao_blur_bg: wgpu::BindGroup,

    // ── Post bind group ───────────────────────────────────────────────────────
    post_bgl: wgpu::BindGroupLayout,
    post_bg: wgpu::BindGroup,

    // ── Shared sampler ────────────────────────────────────────────────────────
    linear_sampler: wgpu::Sampler,

    // ── Phase 5: picking ─────────────────────────────────────────────────────
    picker: Picker,
    /// Maps sphere instance index (0-based) → (object_name, atom_index)
    sphere_instance_map: Vec<AtomRef>,

    /// Ghost spheres: invisible in main pass, used for Ribbon/Surface picking.
    ghost_instances: Option<wgpu::Buffer>,
    ghost_instance_count: u32,
    ghost_instance_map: Vec<AtomRef>,

    /// Per-object residue_id arrays: maps atom index → residue identifier.
    residue_ids_cache: HashMap<String, Vec<u32>>,

    /// Currently highlighted residue_id (0 = no highlight).
    picked_residue_id: u32,

    pub bg_color: wgpu::Color,

    /// Light 1 intensity multiplier (default 1.0).
    pub light_intensity: f32,
    /// Light 1 elevation angle in degrees above the horizontal (default 30.0).
    pub light_elevation_deg: f32,
    /// Light 1 azimuth angle in degrees clockwise from forward (default 20.0).
    pub light_azimuth_deg: f32,
    /// Light 2 intensity multiplier (default 0.0 = off).
    pub light2_intensity: f32,
    /// Light 2 elevation angle in degrees (default -20.0).
    pub light2_elevation_deg: f32,
    /// Light 2 azimuth angle in degrees (default -160.0, roughly opposite to light 1).
    pub light2_azimuth_deg: f32,
    /// Surface transparency alpha (default 0.65). Set via `set transparency`.
    pub surface_alpha: f32,
    /// Edge darkening strength (default 1.0, 0=off). Set via `set edge_strength`.
    pub edge_strength: f32,
    /// PBR roughness (default 0.4, 0=mirror, 1=fully diffuse). Set via `set roughness`.
    pub roughness: f32,
    /// PBR metallic factor (default 0.0). Set via `set metallic`.
    pub metallic: f32,
    /// IBL sky hemisphere color (default soft blue).
    pub sky_color: glam::Vec3,
    /// IBL ground hemisphere color (default dark warm).
    pub ground_color: glam::Vec3,
    /// IBL overall intensity multiplier (default 1.0). Set via `set ibl_intensity`.
    pub ibl_intensity: f32,
    /// Shadow strength (0=no shadow, 1=full shadow). Default 0.4. Set via `set shadow_strength`.
    pub shadow_strength: f32,
    /// Bloom threshold (luminance above which pixels glow). Default 1.0. Set via `set bloom_threshold`.
    pub bloom_threshold: f32,
    /// Bloom intensity multiplier. Default 0.15. Set via `set bloom_intensity`.
    pub bloom_intensity: f32,
    /// Surface computation method (Gaussian or SES). Default Gaussian.
    pub surface_type: crate::render::surface::SurfaceType,
    /// Surface grid step size in Å (default 0.5, smaller = finer mesh). Set via `set surface_quality`.
    pub surface_quality: f32,
    /// Taubin smoothing iterations for the surface mesh (0 = off). Set via `set surface_smooth`.
    pub surface_smooth: u32,
    /// When true (Pocket Surface preset), keep only the surface facing the ligand.
    pub surface_clip_to_ligand: bool,
    /// Ligand–protein hydrogen bonds (heavy-atom endpoint pairs) drawn as dashed
    /// lines in the Binding Site view. Empty in all other presets.
    pub hbond_segments: Vec<(glam::Vec3, glam::Vec3)>,

    // ── Shadow mapping ───────────────────────────────────────────────────────
    shadow_map_view: wgpu::TextureView,
    shadow_uniform_buffer: wgpu::Buffer,
    shadow_uniform_bg: wgpu::BindGroup,
    shadow_bg: wgpu::BindGroup,            // group 1 for main shaders
    shadow_impostor_pipeline: wgpu::RenderPipeline,
    shadow_cylinder_pipeline: wgpu::RenderPipeline,
    shadow_mesh_pipeline: wgpu::RenderPipeline,
    scene_center: glam::Vec3,
    scene_radius: f32,

    // ── Bloom ────────────────────────────────────────────────────────────────
    bloom_down_pipeline: wgpu::RenderPipeline,
    bloom_blur_h_pipeline: wgpu::RenderPipeline,
    bloom_blur_v_pipeline: wgpu::RenderPipeline,
    bloom_down_bgl: wgpu::BindGroupLayout,
    bloom_down_bg: wgpu::BindGroup,
    bloom_blur_bgl: wgpu::BindGroupLayout,
    bloom_blur_h_bg: wgpu::BindGroup,   // reads bloom_a, writes bloom_b
    bloom_blur_v_bg: wgpu::BindGroup,   // reads bloom_b, writes bloom_a
    bloom_a_tex: wgpu::Texture,
    bloom_a_view: wgpu::TextureView,
    bloom_b_tex: wgpu::Texture,
    bloom_b_view: wgpu::TextureView,

    /// egui overlay renderer.
    pub egui_renderer: egui_wgpu::Renderer,

    /// When set, the next render will capture the post-composite output to this path.
    pub pending_screenshot: Option<std::path::PathBuf>,
}

impl RenderState {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });
        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // MSAA depth (multisampled) — also has TEXTURE_BINDING for depth resolve
        let (depth_texture, depth_view) = create_depth_texture(&device, &config, 4);
        // MSAA color (Rgba16Float × 4) — resolve target is scene_color_tex
        let (msaa_texture, msaa_color_view) = create_msaa_color_texture(&device, &config);
        // Single-sample opaque scene resolve target
        let (scene_color_tex, scene_color_view) = create_rgba16float_texture(
            &device, &config, 1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "SceneColor",
        );
        // Single-sample depth for SSAO / post
        let (depth_single_tex, depth_single_view) = create_depth_single_texture(&device, &config);
        // SSAO texture
        let (ssao_tex, ssao_view) = create_r8unorm_texture(&device, &config);

        // ── Shared sampler ───────────────────────────────────────────────────
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("LinearSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ── Shadow map ─────────────────────────────────────────────────────
        let shadow_map_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ShadowMapTex"),
            size: wgpu::Extent3d { width: SHADOW_MAP_SIZE, height: SHADOW_MAP_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_map_view = shadow_map_tex.create_view(&Default::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ShadowSampler"),
            compare: Some(wgpu::CompareFunction::Less),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ── Shadow uniform buffer ───────────────────────────────────────────
        let shadow_uniforms = ShadowUniforms {
            light_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            light_right: [1.0, 0.0, 0.0], _pad0: 0.0,
            light_up:    [0.0, 1.0, 0.0], _pad1: 0.0,
            light_forward: [0.0, 0.0, -1.0], _pad2: 0.0,
        };
        let shadow_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShadowUniforms"),
            contents: bytemuck::bytes_of(&shadow_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shadow_uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowUniformBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let shadow_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShadowUniformBG"),
            layout: &shadow_uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_uniform_buffer.as_entire_binding(),
            }],
        });

        // ── Shadow bind group layout (group 1 for main shaders) ─────────────
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let shadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShadowBG"),
            layout: &shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&shadow_map_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
            ],
        });

        // ── Uniform buffer ───────────────────────────────────────────────────
        let screen_size = [config.width as f32, config.height as f32];
        let uniforms = Uniforms::new(
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            glam::Vec3::new(1.0, 1.0, 1.0),
            glam::Vec3::new(0.0, 0.0, 5.0),
            0,
            1.0,
            screen_size,
            0.65,
            1.0,
            [0.0, 0.0, 0.0],
            glam::Vec3::X,
            glam::Vec3::Y,
            0.4,
            0.0,
            glam::Vec3::new(0.55, 0.65, 0.85),
            1.0,
            glam::Vec3::new(0.15, 0.12, 0.10),
            0.4,
            glam::Mat4::IDENTITY,
            1.0,   // bloom_threshold
            0.0,   // bloom_intensity (off by default)
            glam::Vec3::ZERO, // light2_dir
            0.0,              // light2_intensity (off by default)
        );
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UniformBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UniformBG"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &shadow_bgl],
            push_constant_ranges: &[],
        });

        // ── Sphere impostor pipeline (Rgba16Float, MSAA×4) ───────────────────
        // Billboard quads: 6 vertices per instance, no mesh vertex buffer.
        // Fragment shader performs ray-sphere intersection for correct depth/normal.
        let sphere_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SphereImpostorShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/impostor.wgsl").into()),
        });
        let sphere_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SphereImpostorPipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sphere_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[SphereInstance::impostor_desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sphere_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // billboard always faces camera; ray test handles misses
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: 4, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // ── Cylinder pipeline (Rgba16Float, MSAA×4) ──────────────────────────
        let cyl_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CylShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cylinder.wgsl").into()),
        });
        let cylinder_pipeline = build_pipeline(
            &device,
            &pipeline_layout,
            &cyl_shader,
            "vs_main",
            "fs_main",
            &[Vertex::desc(), CylinderInstance::desc()],
            wgpu::TextureFormat::Rgba16Float,
            4,
        );
        let (c_verts, c_indices) = crate::render::ball_stick::gen_cylinder(32);
        let cylinder_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CylVB"),
            contents: bytemuck::cast_slice(&c_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cylinder_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CylIB"),
            contents: bytemuck::cast_slice(&c_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let cylinder_index_count = c_indices.len() as u32;

        // ── Ligand overlay pipelines (single-sample, drawn after surface) ────
        let ligand_sphere_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("LigandSphereImpostorPipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sphere_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[SphereInstance::impostor_desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sphere_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });
        let ligand_cylinder_pipeline = build_pipeline(
            &device,
            &pipeline_layout,
            &cyl_shader,
            "vs_main",
            "fs_main",
            &[Vertex::desc(), CylinderInstance::desc()],
            wgpu::TextureFormat::Rgba16Float,
            1,
        );

        // ── Ribbon pipeline (Rgba16Float, MSAA×4) ────────────────────────────
        let ribbon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RibbonShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ribbon.wgsl").into()),
        });
        let ribbon_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RibbonPipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ribbon_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[RibbonVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ribbon_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // render both sides
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: 4, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // ── Surface pipeline (alpha-blend into single-sample scene_color_tex) ──
        // Rendered AFTER depth resolve + SSAO, so SSAO and Sobel edge only see
        // opaque geometry depth — the surface gets no dark outlines or AO.
        // depth_write_enabled: false → depth_single_tex stays opaque-only for Post.
        let surface_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SurfaceShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/surface.wgsl").into()),
        });
        let surface_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SurfacePipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &surface_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[RibbonVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &surface_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,              // MC triangles have inconsistent winding
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,    // write depth so nearest surface wins
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // ── Shadow pipelines ─────────────────────────────────────────────────
        let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ShadowPipelineLayout"),
            bind_group_layouts: &[&shadow_uniform_bgl],
            push_constant_ranges: &[],
        });

        // Shadow impostor (sphere billboards from light POV, ray-sphere depth)
        let shadow_imp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShadowImpostorShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shadow_impostor.wgsl").into()),
        });
        let shadow_impostor_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ShadowImpostorPipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_imp_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SphereInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                        wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32   },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shadow_imp_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Shadow cylinder (Rodrigues rotation, depth-only)
        let shadow_cyl_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShadowCylinderShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shadow_cylinder.wgsl").into()),
        });
        let shadow_cylinder_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ShadowCylinderPipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_cyl_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<CylinderInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute { offset: 0,  shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
                            wgpu::VertexAttribute { offset: 16, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
                        ],
                    },
                ],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Shadow mesh (ribbon / surface — simple position transform, depth-only)
        let shadow_mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShadowMeshShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shadow_mesh.wgsl").into()),
        });
        let shadow_mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ShadowMeshPipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_mesh_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<RibbonVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                    ],
                }],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // MC mesh has inconsistent winding
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ── Depth resolve pipeline ────────────────────────────────────────────
        let depth_resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DepthResolveBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: true,
                },
                count: None,
            }],
        });
        let depth_resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DepthResolvePipelineLayout"),
            bind_group_layouts: &[&depth_resolve_bgl],
            push_constant_ranges: &[],
        });
        let depth_resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DepthResolveShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/depth_resolve.wgsl").into()),
        });
        let depth_resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DepthResolvePipeline"),
            layout: Some(&depth_resolve_layout),
            vertex: wgpu::VertexState {
                module: &depth_resolve_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_resolve_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        let depth_resolve_bg = create_depth_resolve_bg(&device, &depth_resolve_bgl, &depth_view);

        // ── SSAO pipeline ─────────────────────────────────────────────────────
        let ssao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAOBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let ssao_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSAOPipelineLayout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &ssao_bgl],
            push_constant_ranges: &[],
        });
        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAOShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ssao.wgsl").into()),
        });
        let ssao_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAOPipeline"),
            layout: Some(&ssao_layout),
            vertex: wgpu::VertexState {
                module: &ssao_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        let ssao_bg = create_ssao_bg(&device, &ssao_bgl, &depth_single_view, &linear_sampler);

        // ── SSAO blur pipeline ────────────────────────────────────────────────
        let (ssao_blur_tex, ssao_blur_view) = create_r8unorm_texture(&device, &config);
        let ssao_blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAOBlurBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let ssao_blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSAOBlurPipelineLayout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &ssao_blur_bgl],
            push_constant_ranges: &[],
        });
        let ssao_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAOBlurShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ssao_blur.wgsl").into()),
        });
        let ssao_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAOBlurPipeline"),
            layout: Some(&ssao_blur_layout),
            vertex: wgpu::VertexState {
                module: &ssao_blur_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_blur_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });
        let ssao_blur_bg = create_ssao_blur_bg(
            &device, &ssao_blur_bgl, &ssao_view, &depth_single_view, &linear_sampler,
        );

        // ── Bloom pipelines ─────────────────────────────────────────────────
        let bloom_half_w = (config.width / 2).max(1);
        let bloom_half_h = (config.height / 2).max(1);
        let (bloom_a_tex, bloom_a_view) = create_bloom_texture(&device, bloom_half_w, bloom_half_h, "BloomA");
        let (bloom_b_tex, bloom_b_view) = create_bloom_texture(&device, bloom_half_w, bloom_half_h, "BloomB");

        // Bloom downsample BGL: reads scene_color (full-res), writes bright to half-res
        let bloom_down_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BloomDownBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bloom_down_bg = create_bloom_down_bg(&device, &bloom_down_bgl, &scene_color_view, &linear_sampler);

        let bloom_down_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BloomDownLayout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &bloom_down_bgl],
            push_constant_ranges: &[],
        });
        let bloom_down_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BloomDownShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bloom_down.wgsl").into()),
        });
        let bloom_down_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("BloomDownPipeline"),
            layout: Some(&bloom_down_layout),
            vertex: wgpu::VertexState {
                module: &bloom_down_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_down_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // Bloom blur BGL: reads one bloom tex, writes the other
        let bloom_blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BloomBlurBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bloom_blur_h_bg = create_bloom_blur_bg(&device, &bloom_blur_bgl, &bloom_a_view, &linear_sampler, "BloomBlurH_BG");
        let bloom_blur_v_bg = create_bloom_blur_bg(&device, &bloom_blur_bgl, &bloom_b_view, &linear_sampler, "BloomBlurV_BG");

        let bloom_blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BloomBlurLayout"),
            bind_group_layouts: &[&bloom_blur_bgl],
            push_constant_ranges: &[],
        });
        let bloom_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BloomBlurShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bloom_blur.wgsl").into()),
        });
        let bloom_blur_h_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("BloomBlurH"),
            layout: Some(&bloom_blur_layout),
            vertex: wgpu::VertexState {
                module: &bloom_blur_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_blur_shader,
                entry_point: Some("fs_blur_h"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });
        let bloom_blur_v_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("BloomBlurV"),
            layout: Some(&bloom_blur_layout),
            vertex: wgpu::VertexState {
                module: &bloom_blur_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_blur_shader,
                entry_point: Some("fs_blur_v"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // ── Post pipeline ─────────────────────────────────────────────────────
        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostBGL"),
            entries: &[
                // scene_tex (binding 0)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // ssao_tex (binding 1)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // depth_tex (binding 2)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // lin_samp (binding 3)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // bloom_tex (binding 4)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let post_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostPipelineLayout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &post_bgl],
            push_constant_ranges: &[],
        });
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PostShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/post.wgsl").into()),
        });
        let post_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PostPipeline"),
            layout: Some(&post_layout),
            vertex: wgpu::VertexState {
                module: &post_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &post_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        let post_bg = create_post_bg(
            &device, &post_bgl,
            &scene_color_view, &ssao_blur_view, &depth_single_view, &linear_sampler,
            &bloom_a_view,
        );

        // ── Phase 5: picker ──────────────────────────────────────────────────
        let picker = Picker::new(&device, size.width.max(1), size.height.max(1), &uniform_bind_group_layout);

        // ── egui renderer ────────────────────────────────────────────────────
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            uniform_buffer,
            uniform_bind_group,
            uniform_bind_group_layout,
            sphere_pipeline,
            sphere_instances: None,
            sphere_instance_count: 0,
            ligand_sphere_pipeline,
            ligand_cylinder_pipeline,
            ligand_sphere_instances: None,
            ligand_sphere_instance_count: 0,
            ligand_cylinder_instances: None,
            ligand_cylinder_instance_count: 0,
            cylinder_pipeline,
            cylinder_vb,
            cylinder_ib,
            cylinder_index_count,
            cylinder_instances: None,
            cylinder_instance_count: 0,
            ribbon_pipeline,
            ribbon_vb: None,
            ribbon_ib: None,
            ribbon_index_count: 0,
            surface_pipeline,
            surface_vb: None,
            surface_ib: None,
            surface_index_count: 0,
            depth_texture,
            depth_view,
            msaa_texture,
            msaa_color_view,
            scene_color_tex,
            scene_color_view,
            depth_single_tex,
            depth_single_view,
            ssao_tex,
            ssao_view,
            depth_resolve_pipeline,
            ssao_pipeline,
            post_pipeline,
            depth_resolve_bgl,
            depth_resolve_bg,
            ssao_bgl,
            ssao_bg,
            ssao_blur_tex,
            ssao_blur_view,
            ssao_blur_pipeline,
            ssao_blur_bgl,
            ssao_blur_bg,
            post_bgl,
            post_bg,
            linear_sampler,
            picker,
            sphere_instance_map: Vec::new(),
            ghost_instances: None,
            ghost_instance_count: 0,
            ghost_instance_map: Vec::new(),
            residue_ids_cache: HashMap::new(),
            picked_residue_id: 0,
            bg_color: wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            light_intensity: 1.0,
            light_elevation_deg: 30.0,
            light_azimuth_deg: 20.0,
            light2_intensity: 0.0,
            light2_elevation_deg: -20.0,
            light2_azimuth_deg: -160.0,
            surface_alpha: 1.0,
            edge_strength: 1.0,
            roughness: 0.4,
            metallic: 0.0,
            sky_color:    glam::Vec3::new(0.55, 0.65, 0.85),
            ground_color: glam::Vec3::new(0.15, 0.12, 0.10),
            ibl_intensity: 1.0,
            shadow_strength: 0.4,
            bloom_threshold: 1.0,
            bloom_intensity: 0.0,
            surface_type: crate::render::surface::SurfaceType::Ses,
            surface_quality: 0.35,
            surface_smooth: 6,
            surface_clip_to_ligand: false,
            hbond_segments: Vec::new(),
            shadow_map_view,
            shadow_uniform_buffer,
            shadow_uniform_bg,
            shadow_bg,
            shadow_impostor_pipeline,
            shadow_cylinder_pipeline,
            shadow_mesh_pipeline,
            scene_center: glam::Vec3::ZERO,
            scene_radius: 50.0,
            bloom_down_pipeline,
            bloom_blur_h_pipeline,
            bloom_blur_v_pipeline,
            bloom_down_bgl,
            bloom_down_bg,
            bloom_blur_bgl,
            bloom_blur_h_bg,
            bloom_blur_v_bg,
            bloom_a_tex,
            bloom_a_view,
            bloom_b_tex,
            bloom_b_view,
            egui_renderer,
            pending_screenshot: None,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);

        let (dt, dv) = create_depth_texture(&self.device, &self.config, 4);
        self.depth_texture = dt;
        self.depth_view = dv;

        let (mt, mv) = create_msaa_color_texture(&self.device, &self.config);
        self.msaa_texture = mt;
        self.msaa_color_view = mv;

        let (sct, scv) = create_rgba16float_texture(
            &self.device, &self.config, 1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            "SceneColor",
        );
        self.scene_color_tex = sct;
        self.scene_color_view = scv;

        let (dst, dsv) = create_depth_single_texture(&self.device, &self.config);
        self.depth_single_tex = dst;
        self.depth_single_view = dsv;

        let (st, sv) = create_r8unorm_texture(&self.device, &self.config);
        self.ssao_tex = st;
        self.ssao_view = sv;

        let (sbt, sbv) = create_r8unorm_texture(&self.device, &self.config);
        self.ssao_blur_tex = sbt;
        self.ssao_blur_view = sbv;

        // Recreate bind groups (views have changed)
        self.depth_resolve_bg = create_depth_resolve_bg(&self.device, &self.depth_resolve_bgl, &self.depth_view);
        self.ssao_bg = create_ssao_bg(&self.device, &self.ssao_bgl, &self.depth_single_view, &self.linear_sampler);
        self.ssao_blur_bg = create_ssao_blur_bg(
            &self.device, &self.ssao_blur_bgl,
            &self.ssao_view, &self.depth_single_view, &self.linear_sampler,
        );

        // Bloom textures
        let bloom_half_w = (width / 2).max(1);
        let bloom_half_h = (height / 2).max(1);
        let (bat, bav) = create_bloom_texture(&self.device, bloom_half_w, bloom_half_h, "BloomA");
        self.bloom_a_tex = bat;
        self.bloom_a_view = bav;
        let (bbt, bbv) = create_bloom_texture(&self.device, bloom_half_w, bloom_half_h, "BloomB");
        self.bloom_b_tex = bbt;
        self.bloom_b_view = bbv;
        self.bloom_down_bg = create_bloom_down_bg(&self.device, &self.bloom_down_bgl, &self.scene_color_view, &self.linear_sampler);
        self.bloom_blur_h_bg = create_bloom_blur_bg(&self.device, &self.bloom_blur_bgl, &self.bloom_a_view, &self.linear_sampler, "BloomBlurH_BG");
        self.bloom_blur_v_bg = create_bloom_blur_bg(&self.device, &self.bloom_blur_bgl, &self.bloom_b_view, &self.linear_sampler, "BloomBlurV_BG");

        self.post_bg = create_post_bg(
            &self.device, &self.post_bgl,
            &self.scene_color_view, &self.ssao_blur_view, &self.depth_single_view, &self.linear_sampler,
            &self.bloom_a_view,
        );

        self.picker.resize(&self.device, width, height);
    }

    /// Rebuild GPU geometry buffers from scene data.
    ///
    /// Only the parts indicated by `dirty` are rebuilt:
    /// - `ATOMS` / `RIBBON`: spheres, cylinders, backbone, lines, ribbon mesh, ghost spheres
    /// - `SURFACE`: surface mesh (most expensive)
    pub fn upload_scene(&mut self, scene: &Scene, dirty: crate::scene::SceneDirty) {
        use crate::scene::SceneDirty;
        let _upload_t0 = std::time::Instant::now();

        let need_atoms_ribbon = dirty.contains(SceneDirty::ATOMS) || dirty.contains(SceneDirty::RIBBON);
        let need_surface = dirty.contains(SceneDirty::SURFACE);

        // Residue IDs are needed by both ribbon and surface builds.
        self.residue_ids_cache.clear();
        for (obj_name, obj) in scene.iter() {
            if !obj.is_visible() { continue; }
            let residue_ids = compute_residue_ids(&obj.structure);
            self.residue_ids_cache.insert(obj_name.clone(), residue_ids);
        }

        // ── Atoms + Ribbon (share cylinder buffer via ribbon gap lines) ─────
        if need_atoms_ribbon {
            let mut spheres:      Vec<SphereInstance>   = Vec::new();
            let mut sphere_map:   Vec<AtomRef>          = Vec::new();
            let mut cylinders:    Vec<CylinderInstance> = Vec::new();
            let mut ligand_spheres:   Vec<SphereInstance>   = Vec::new();
            let mut ligand_cylinders: Vec<CylinderInstance> = Vec::new();
            let mut ribbon_verts: Vec<RibbonVertex>     = Vec::new();
            let mut ribbon_idxs:  Vec<u32>              = Vec::new();

            for (obj_name, obj) in scene.iter() {
                if !obj.is_visible() { continue; }
                let atoms  = &obj.structure.atoms;
                let colors = &obj.atom_colors;

                // ── Ball-and-stick ────────────────────────────────────────────
                for (i, atom) in atoms.iter().enumerate() {
                    let flags = obj.atom_rep_show.get(i).copied().unwrap_or(0);
                    if flags & (REP_BALL_STICK | REP_STICK) == 0 {
                        continue;
                    }
                    // Stick: uniform bond-radius spheres round the joints into a
                    // continuous rod. Ball-and-stick keeps van-der-Waals-scaled balls.
                    let stick_only = flags & REP_BALL_STICK == 0;
                    let is_water = atom.is_hetatm
                        && matches!(atom.residue.name.as_str(), "HOH" | "WAT" | "DOD");
                    let is_ligand = atom.is_hetatm && !is_water;
                    let color  = colors[i];
                    let radius = if stick_only {
                        STICK_RADIUS
                    } else {
                        vdw_radius(&atom.element) * if is_water { 0.14 } else { 0.22 }
                    };
                    let edge_boost = if is_ligand { 1.0 } else { 0.0 };
                    let inst = SphereInstance { position: atom.position.to_array(), radius, color, edge_boost };
                    sphere_map.push((obj_name.clone(), i));
                    if is_ligand {
                        ligand_spheres.push(inst);
                    } else {
                        spheres.push(inst);
                    }
                }
                for bond in &obj.structure.bonds {
                    let (a1, a2) = (bond.atom1, bond.atom2);
                    if a1 >= atoms.len() || a2 >= atoms.len() { continue; }
                    let f1 = obj.atom_rep_show.get(a1).copied().unwrap_or(0);
                    let f2 = obj.atom_rep_show.get(a2).copied().unwrap_or(0);
                    let mask = REP_BALL_STICK | REP_STICK;
                    if f1 & mask == 0 || f2 & mask == 0 { continue; }
                    if atoms[a1].is_hetatm != atoms[a2].is_hetatm {
                        let same_residue = atoms[a1].residue.chain == atoms[a2].residue.chain
                            && atoms[a1].residue.seq_num == atoms[a2].residue.seq_num
                            && atoms[a1].residue.ins_code == atoms[a2].residue.ins_code;
                        if !same_residue { continue; }
                    }
                    let p1  = atoms[a1].position;
                    let p2  = atoms[a2].position;
                    let mid = (p1 + p2) * 0.5;
                    let is_ligand_a1 = atoms[a1].is_hetatm && !matches!(atoms[a1].residue.name.as_str(), "HOH" | "WAT" | "DOD");
                    let is_ligand_a2 = atoms[a2].is_hetatm && !matches!(atoms[a2].residue.name.as_str(), "HOH" | "WAT" | "DOD");
                    let eb1 = if is_ligand_a1 { 1.0 } else { 0.0 };
                    let eb2 = if is_ligand_a2 { 1.0 } else { 0.0 };
                    // Stick bonds (both atoms stick-only, no ball) are thicker.
                    let stick_bond = (f1 & REP_BALL_STICK == 0) && (f2 & REP_BALL_STICK == 0);
                    let radius = if stick_bond { STICK_RADIUS } else { BOND_RADIUS };
                    let c1 = CylinderInstance::new(p1.to_array(), mid.to_array(), radius, colors[a1], eb1);
                    let c2 = CylinderInstance::new(mid.to_array(), p2.to_array(), radius, colors[a2], eb2);
                    if is_ligand_a1 || is_ligand_a2 {
                        ligand_cylinders.push(c1);
                        ligand_cylinders.push(c2);
                    } else {
                        cylinders.push(c1);
                        cylinders.push(c2);
                    }
                }

                // ── Aromatic ring indicators (HETATM ligands only, dashed) ─────
                if obj.has_representation(RepresentationType::BallAndStick)
                    && atoms.iter().any(|a| a.is_hetatm)
                {
                    let rings = detect_aromatic_rings(&obj.structure);
                    for ring in &rings {
                        if !ring.atom_indices.iter().all(|&i| atoms[i].is_hetatm) {
                            continue;
                        }
                        let avg_color = {
                            let mut c = [0.0f32; 3];
                            let n = ring.atom_indices.len() as f32;
                            for &idx in &ring.atom_indices {
                                let ac = colors[idx];
                                c[0] += ac[0]; c[1] += ac[1]; c[2] += ac[2];
                            }
                            [c[0] / n, c[1] / n, c[2] / n]
                        };
                        // Circle radius scales with the ring size (avg distance
                        // from atoms to center → smaller for 5-membered rings).
                        let avg_dist = ring.atom_indices.iter()
                            .map(|&i| (atoms[i].position - ring.center).length())
                            .sum::<f32>() / ring.atom_indices.len() as f32;
                        let r = avg_dist * AROMATIC_RING_SCALE;

                        let u = ring.normal.cross(glam::Vec3::Y).normalize_or_zero();
                        let u = if u.length_squared() < 0.5 {
                            ring.normal.cross(glam::Vec3::X).normalize()
                        } else { u };
                        let v = ring.normal.cross(u).normalize();

                        // Dashed circle: draw every other segment as a thin cylinder.
                        for seg in 0..AROMATIC_RING_SEGMENTS {
                            if seg % 2 != 0 { continue; }
                            let a0 = std::f32::consts::TAU * seg as f32 / AROMATIC_RING_SEGMENTS as f32;
                            let a1 = std::f32::consts::TAU * (seg + 1) as f32 / AROMATIC_RING_SEGMENTS as f32;
                            let p0 = ring.center + u * (r * a0.cos()) + v * (r * a0.sin());
                            let p1 = ring.center + u * (r * a1.cos()) + v * (r * a1.sin());
                            ligand_cylinders.push(CylinderInstance::new(
                                p0.to_array(), p1.to_array(),
                                AROMATIC_RING_RADIUS, avg_color, 1.0,
                            ));
                        }
                    }
                }

                // ── Ribbon ───────────────────────────────────────────────────
                if obj.has_representation(RepresentationType::Ribbon) {
                    let rids = self.residue_ids_cache.get(obj_name).map(|v| v.as_slice()).unwrap_or(&[]);
                    let verts_start = ribbon_verts.len();
                    let mut ribbon_gaps: Vec<RibbonGap> = Vec::new();
                    build_ribbon(&obj.structure, &obj.atom_colors, rids, &obj.atom_rep_show, &mut ribbon_verts, &mut ribbon_idxs, &mut ribbon_gaps);
                    if let Some(col) = obj.ribbon_color_override {
                        for v in &mut ribbon_verts[verts_start..] {
                            v.color = col;
                        }
                    }
                    for gap in &ribbon_gaps {
                        emit_dashed_cylinders(&mut cylinders, &gap.p1, &gap.p2, &gap.color1, &gap.color2);
                    }
                }

                // ── Backbone (Cα trace) ──────────────────────────────────────
                if obj.has_representation(RepresentationType::Backbone) {
                    let mut ca_by_chain: HashMap<char, Vec<(i32, Option<char>, usize)>> =
                        HashMap::new();
                    for (i, atom) in atoms.iter().enumerate() {
                        if obj.atom_rep_show.get(i).copied().unwrap_or(0) & REP_BACKBONE == 0 { continue; }
                        if atom.name.trim() == "CA" && !atom.is_hetatm {
                            ca_by_chain
                                .entry(atom.residue.chain)
                                .or_default()
                                .push((atom.residue.seq_num, atom.residue.ins_code, i));
                        }
                    }
                    for chain_cas in ca_by_chain.values_mut() {
                        chain_cas.sort_unstable_by_key(|&(seq, ins, _)| (seq, ins));
                        for &(_, _, i) in chain_cas.iter() {
                            sphere_map.push((obj_name.clone(), i));
                            spheres.push(SphereInstance {
                                position: atoms[i].position.to_array(),
                                radius: BACKBONE_JOINT_RADIUS,
                                color: colors[i],
                                edge_boost: 0.0,
                            });
                        }
                        for window in chain_cas.windows(2) {
                            let (seq1, _, i1) = window[0];
                            let (seq2, _, i2) = window[1];
                            let p1  = atoms[i1].position.to_array();
                            let p2  = atoms[i2].position.to_array();
                            if residues_consecutive(seq1, seq2) {
                                let mid = [(p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5, (p1[2]+p2[2])*0.5];
                                cylinders.push(CylinderInstance::new(p1,  mid, BACKBONE_TUBE_RADIUS, colors[i1], 0.0));
                                cylinders.push(CylinderInstance::new(mid, p2,  BACKBONE_TUBE_RADIUS, colors[i2], 0.0));
                            } else {
                                emit_dashed_cylinders(&mut cylinders, &p1, &p2, &colors[i1], &colors[i2]);
                            }
                        }
                    }
                }

                // ── Lines (wire) ─────────────────────────────────────────────
                const LINE_RADIUS: f32 = 0.04;
                for bond in &obj.structure.bonds {
                    let (a1, a2) = (bond.atom1, bond.atom2);
                    if a1 >= atoms.len() || a2 >= atoms.len() { continue; }
                    let f1 = obj.atom_rep_show.get(a1).copied().unwrap_or(0);
                    let f2 = obj.atom_rep_show.get(a2).copied().unwrap_or(0);
                    if f1 & REP_LINES == 0 || f2 & REP_LINES == 0 { continue; }
                    if atoms[a1].is_hetatm != atoms[a2].is_hetatm {
                        let same_residue = atoms[a1].residue.chain == atoms[a2].residue.chain
                            && atoms[a1].residue.seq_num == atoms[a2].residue.seq_num
                            && atoms[a1].residue.ins_code == atoms[a2].residue.ins_code;
                        if !same_residue { continue; }
                    }
                    let p1  = atoms[a1].position.to_array();
                    let p2  = atoms[a2].position.to_array();
                    let mid = [(p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5, (p1[2]+p2[2])*0.5];
                    cylinders.push(CylinderInstance::new(p1,  mid, LINE_RADIUS, colors[a1], 0.0));
                    cylinders.push(CylinderInstance::new(mid, p2,  LINE_RADIUS, colors[a2], 0.0));
                }
            }

            // ── Ghost spheres for Ribbon / Surface picking ──────────────────
            let mut ghost_spheres: Vec<SphereInstance> = Vec::new();
            let mut ghost_map: Vec<AtomRef> = Vec::new();
            for (obj_name, obj) in scene.iter() {
                if !obj.is_visible() { continue; }
                for (i, atom) in obj.structure.atoms.iter().enumerate() {
                    let flags = obj.atom_rep_show.get(i).copied().unwrap_or(0);
                    let atom_has_ribbon  = flags & REP_RIBBON  != 0;
                    let atom_has_surface = flags & REP_SURFACE != 0;
                    if !atom_has_ribbon && !atom_has_surface { continue; }
                    if matches!(atom.residue.name.as_str(), "HOH" | "WAT" | "DOD") { continue; }
                    if atom_has_ribbon && !atom_has_surface {
                        let name = atom.name.trim();
                        if !matches!(name, "N" | "CA" | "C" | "O") { continue; }
                    }
                    ghost_map.push((obj_name.clone(), i));
                    ghost_spheres.push(SphereInstance {
                        position: atom.position.to_array(),
                        radius: vdw_radius(&atom.element),
                        color: [0.0, 0.0, 0.0],
                        edge_boost: 0.0,
                    });
                }
            }

            // ── Ligand–protein hydrogen bonds (dashed lines, Binding Site) ────
            {
                const HBOND_COLOR:  [f32; 3] = [0.98, 0.86, 0.25]; // soft yellow
                const HBOND_RADIUS: f32 = 0.055;
                const DASH:         f32 = 0.28;   // dash length; period = 2×DASH
                for &(a, b) in &self.hbond_segments {
                    let total = (b - a).length();
                    if total < 1e-4 { continue; }
                    let dir = (b - a) / total;
                    let mut t = 0.0;
                    while t < total {
                        let p0 = a + dir * t;
                        let p1 = a + dir * (t + DASH).min(total);
                        cylinders.push(CylinderInstance::new(
                            p0.to_array(), p1.to_array(), HBOND_RADIUS, HBOND_COLOR, 0.0,
                        ));
                        t += DASH * 2.0;          // skip the gap
                    }
                }
            }

            // Upload atom/ribbon/ghost buffers
            self.ghost_instance_map = ghost_map;
            self.ghost_instance_count = ghost_spheres.len() as u32;
            self.ghost_instances = if ghost_spheres.is_empty() {
                None
            } else {
                Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("GhostInstances"),
                    contents: bytemuck::cast_slice(&ghost_spheres),
                    usage: wgpu::BufferUsages::VERTEX,
                }))
            };

            self.sphere_instance_map = sphere_map;
            self.sphere_instance_count = spheres.len() as u32;
            self.sphere_instances = if spheres.is_empty() {
                None
            } else {
                Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SphereInstances"),
                    contents: bytemuck::cast_slice(&spheres),
                    usage: wgpu::BufferUsages::VERTEX,
                }))
            };

            self.cylinder_instance_count = cylinders.len() as u32;
            self.cylinder_instances = if cylinders.is_empty() {
                None
            } else {
                Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("CylInstances"),
                    contents: bytemuck::cast_slice(&cylinders),
                    usage: wgpu::BufferUsages::VERTEX,
                }))
            };

            self.ligand_sphere_instance_count = ligand_spheres.len() as u32;
            self.ligand_sphere_instances = if ligand_spheres.is_empty() {
                None
            } else {
                Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("LigandSphereInstances"),
                    contents: bytemuck::cast_slice(&ligand_spheres),
                    usage: wgpu::BufferUsages::VERTEX,
                }))
            };

            self.ligand_cylinder_instance_count = ligand_cylinders.len() as u32;
            self.ligand_cylinder_instances = if ligand_cylinders.is_empty() {
                None
            } else {
                Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("LigandCylInstances"),
                    contents: bytemuck::cast_slice(&ligand_cylinders),
                    usage: wgpu::BufferUsages::VERTEX,
                }))
            };

            self.ribbon_index_count = ribbon_idxs.len() as u32;
            if ribbon_verts.is_empty() {
                self.ribbon_vb = None;
                self.ribbon_ib = None;
            } else {
                self.ribbon_vb = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("RibbonVB"),
                    contents: bytemuck::cast_slice(&ribbon_verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }));
                self.ribbon_ib = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("RibbonIB"),
                    contents: bytemuck::cast_slice(&ribbon_idxs),
                    usage: wgpu::BufferUsages::INDEX,
                }));
            }

            log::info!(
                "upload_scene [atoms+ribbon]: {:.0} ms  (spheres={}, cyls={}, ribbon_tris={})",
                _upload_t0.elapsed().as_secs_f64() * 1000.0,
                spheres.len(),
                cylinders.len(),
                ribbon_idxs.len() / 3,
            );
        }

        // ── Surface ─────────────────────────────────────────────────────────
        if need_surface {
            let _surf_t0 = std::time::Instant::now();
            let mut surface_verts: Vec<RibbonVertex> = Vec::new();
            let mut surface_idxs:  Vec<u32>          = Vec::new();

            for (obj_name, obj) in scene.iter() {
                if !obj.is_visible() { continue; }
                if obj.has_representation(RepresentationType::Surface) {
                    let rids = self.residue_ids_cache.get(obj_name).map(|v| v.as_slice()).unwrap_or(&[]);
                    let verts_start = surface_verts.len();
                    build_surface(&obj.structure, &obj.atom_colors, rids, &obj.atom_rep_show, self.surface_type, self.surface_quality, self.surface_smooth as usize, &mut surface_verts, &mut surface_idxs);
                    if let Some(col) = obj.surface_color_override {
                        for v in &mut surface_verts[verts_start..] {
                            v.color = col;
                        }
                    }
                    log::info!(
                        "surface build '{}': {:.0} ms  ({} verts, {} tris)",
                        obj_name,
                        _surf_t0.elapsed().as_secs_f64() * 1000.0,
                        surface_verts.len(),
                        surface_idxs.len() / 3,
                    );
                }
            }

            // ── Pocket mode: keep only the ligand-facing side of the surface ──
            // Collect ligand atoms (non-polymer, non-water) across the scene and
            // drop every surface triangle whose outward normal points away from
            // the nearest ligand atom, so only the pocket wall remains.
            if self.surface_clip_to_ligand && !surface_verts.is_empty() {
                let anchors: Vec<glam::Vec3> = scene
                    .iter()
                    .filter(|(_, o)| o.is_visible())
                    .flat_map(|(_, o)| {
                        o.structure.atoms.iter().filter(move |a| {
                            !o.structure.is_polymer_atom(a)
                                && !matches!(a.residue.name.as_str(), "HOH" | "WAT" | "DOD")
                        })
                    })
                    .map(|a| a.position)
                    .collect();

                if !anchors.is_empty() {
                    // Per-vertex facing value: outward normal · direction to the
                    // nearest ligand atom (+1 = straight at the ligand).
                    let facing: Vec<f32> = surface_verts
                        .iter()
                        .map(|v| {
                            let p = glam::Vec3::from(v.position);
                            let n = glam::Vec3::from(v.normal);
                            let nearest = anchors
                                .iter()
                                .copied()
                                .min_by(|a, b| {
                                    (*a - p).length_squared().total_cmp(&(*b - p).length_squared())
                                })
                                .unwrap();
                            let dir = nearest - p;
                            if dir.length_squared() < 1e-6 { 1.0 } else { n.dot(dir.normalize()) }
                        })
                        .collect();

                    // Keep a triangle when its vertices face the ligand on average.
                    let kept: Vec<u32> = surface_idxs
                        .chunks(3)
                        .filter(|t| {
                            let avg = (facing[t[0] as usize]
                                + facing[t[1] as usize]
                                + facing[t[2] as usize]) / 3.0;
                            avg > SURFACE_POCKET_FACING
                        })
                        .flat_map(|t| t.iter().copied())
                        .collect();

                    // Prune isolated fragments left behind (e.g. stray back-side bits).
                    let pruned = drop_small_components(
                        surface_verts.len(),
                        kept,
                        SURFACE_POCKET_MIN_COMPONENT,
                    );
                    // Fill the small holes the clip opened up in the pocket wall.
                    surface_idxs = fill_mesh_holes(pruned, SURFACE_HOLE_MAX_EDGES);
                }
            }

            self.surface_index_count = surface_idxs.len() as u32;
            if surface_verts.is_empty() || surface_idxs.is_empty() {
                self.surface_vb = None;
                self.surface_ib = None;
            } else {
                self.surface_vb = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SurfaceVB"),
                    contents: bytemuck::cast_slice(&surface_verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }));
                self.surface_ib = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SurfaceIB"),
                    contents: bytemuck::cast_slice(&surface_idxs),
                    usage: wgpu::BufferUsages::INDEX,
                }));
            }
        }

        // ── Compute scene bounding sphere for shadow mapping ────────────────
        {
            let mut center = glam::Vec3::ZERO;
            let mut n = 0u32;
            for (_, obj) in scene.iter() {
                if !obj.is_visible() { continue; }
                for atom in &obj.structure.atoms {
                    center += atom.position;
                    n += 1;
                }
            }
            if n > 0 {
                center /= n as f32;
                let mut max_r2 = 0.0f32;
                for (_, obj) in scene.iter() {
                    if !obj.is_visible() { continue; }
                    for atom in &obj.structure.atoms {
                        let d2 = (atom.position - center).length_squared();
                        if d2 > max_r2 { max_r2 = d2; }
                    }
                }
                self.scene_center = center;
                self.scene_radius = max_r2.sqrt() + 5.0; // margin for VdW radii + surface
            }
        }

        log::info!(
            "upload_scene total: {:.0} ms  (dirty={:?})",
            _upload_t0.elapsed().as_secs_f64() * 1000.0,
            dirty,
        );
    }

    pub fn update_uniforms(&self, camera: &Camera) {
        let view  = camera.view_matrix();
        let proj  = camera.projection_matrix();
        let inv_proj = proj.inverse();
        // Compute light direction from elevation/azimuth angles in camera space.
        let az = self.light_azimuth_deg.to_radians();
        let el = self.light_elevation_deg.to_radians();
        let light_base = glam::Vec3::new(
            el.cos() * az.sin(),
            el.sin(),
            el.cos() * az.cos(),
        );
        let light_dir = camera.rotation * light_base;

        // Light 2
        let az2 = self.light2_azimuth_deg.to_radians();
        let el2 = self.light2_elevation_deg.to_radians();
        let light2_base = glam::Vec3::new(
            el2.cos() * az2.sin(),
            el2.sin(),
            el2.cos() * az2.cos(),
        );
        let light2_dir = camera.rotation * light2_base;

        let screen_size = [self.config.width as f32, self.config.height as f32];
        let bg = [
            self.bg_color.r as f32,
            self.bg_color.g as f32,
            self.bg_color.b as f32,
        ];
        let camera_right = camera.rotation * glam::Vec3::X;
        let camera_up    = camera.rotation * glam::Vec3::Y;

        // ── Light matrices for shadow mapping ─────────────────────────────
        let light_dir_n = light_dir.normalize();
        let r = self.scene_radius.max(1.0);
        let light_eye = self.scene_center + light_dir_n * r * 2.0;
        let up_hint = if light_dir_n.y.abs() > 0.99 { glam::Vec3::Z } else { glam::Vec3::Y };
        let light_view = glam::Mat4::look_at_rh(light_eye, self.scene_center, up_hint);
        let light_proj = glam::Mat4::orthographic_rh(-r, r, -r, r, 0.01, r * 4.5);
        let light_view_proj = light_proj * light_view;

        let light_right = light_dir_n.cross(up_hint).normalize();
        let light_up = light_right.cross(light_dir_n).normalize();
        let light_forward = -light_dir_n; // into the scene

        // Update shadow uniforms
        let shadow_u = ShadowUniforms {
            light_view_proj: light_view_proj.to_cols_array_2d(),
            light_right: light_right.to_array(), _pad0: 0.0,
            light_up: light_up.to_array(), _pad1: 0.0,
            light_forward: light_forward.to_array(), _pad2: 0.0,
        };
        self.queue.write_buffer(&self.shadow_uniform_buffer, 0, bytemuck::bytes_of(&shadow_u));

        let uniforms = Uniforms::new(
            proj * view,
            inv_proj,
            light_dir,
            camera.eye_position(),
            self.picked_residue_id,
            self.light_intensity,
            screen_size,
            self.surface_alpha,
            self.edge_strength,
            bg,
            camera_right,
            camera_up,
            self.roughness,
            self.metallic,
            self.sky_color,
            self.ibl_intensity,
            self.ground_color,
            self.shadow_strength,
            light_view_proj,
            self.bloom_threshold,
            self.bloom_intensity,
            light2_dir,
            self.light2_intensity,
        );
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Look up the residue_id for a given (obj_name, atom_idx) pair.
    pub fn get_residue_id(&self, obj_name: &str, atom_idx: usize) -> u32 {
        self.residue_ids_cache
            .get(obj_name)
            .and_then(|ids| ids.get(atom_idx))
            .copied()
            .unwrap_or(0)
    }

    /// Set the highlighted residue (written to GPU on next update_uniforms).
    pub fn set_highlight(&mut self, residue_id: u32) {
        self.picked_residue_id = residue_id;
    }

    /// Clear the highlight (residue_id = 0 means no highlight in shader).
    pub fn clear_highlight(&mut self) {
        self.picked_residue_id = 0;
    }

    /// Perform a color-ID pick at physical pixel (px, py).
    pub fn pick_at(&self, px: u32, py: u32) -> Option<PickResult> {
        // Phase 1: exact hit on render spheres (atom-level).
        if let Some(instances) = &self.sphere_instances {
            if self.sphere_instance_count > 0 {
                if let Some(idx) = self.picker.pick_at(
                    &self.device,
                    &self.queue,
                    &self.uniform_bind_group,
                    instances,
                    self.sphere_instance_count,
                    px,
                    py,
                ) {
                    if let Some(atom_ref) = self.sphere_instance_map.get(idx as usize) {
                        return Some(PickResult::Atom(atom_ref.clone()));
                    }
                }
            }
        }

        // Phase 2: nearest-search on ghost spheres (residue-level).
        if let Some(ghost_inst) = &self.ghost_instances {
            if self.ghost_instance_count > 0 {
                if let Some(idx) = self.picker.pick_nearest(
                    &self.device,
                    &self.queue,
                    &self.uniform_bind_group,
                    ghost_inst,
                    self.ghost_instance_count,
                    px,
                    py,
                ) {
                    if let Some(atom_ref) = self.ghost_instance_map.get(idx as usize) {
                        return Some(PickResult::Residue(atom_ref.clone()));
                    }
                }
            }
        }

        None
    }

    /// Render the 3-D scene and then the egui overlay in one submission.
    pub fn render(
        &mut self,
        egui_primitives: &[egui::ClippedPrimitive],
        screen_desc: &egui_wgpu::ScreenDescriptor,
        textures_delta: egui::TexturesDelta,
    ) -> anyhow::Result<()> {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        // Final sRGB surface texture — post composite and egui render here.
        let output_view = output.texture.create_view(&Default::default());

        // Upload any new egui textures.
        for (id, delta) in &textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Upload egui vertex/index buffers into the encoder.
        self.egui_renderer.update_buffers(
            &self.device, &self.queue, &mut encoder, egui_primitives, screen_desc,
        );

        // ── Pass 0: Shadow map ──────────────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ShadowPass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_map_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Shadow spheres (impostors)
            if let Some(buf) = &self.sphere_instances {
                if self.sphere_instance_count > 0 {
                    pass.set_pipeline(&self.shadow_impostor_pipeline);
                    pass.set_bind_group(0, &self.shadow_uniform_bg, &[]);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..6, 0..self.sphere_instance_count);
                }
            }

            // Shadow cylinders
            if let Some(buf) = &self.cylinder_instances {
                if self.cylinder_instance_count > 0 {
                    pass.set_pipeline(&self.shadow_cylinder_pipeline);
                    pass.set_bind_group(0, &self.shadow_uniform_bg, &[]);
                    pass.set_vertex_buffer(0, self.cylinder_vb.slice(..));
                    pass.set_vertex_buffer(1, buf.slice(..));
                    pass.set_index_buffer(self.cylinder_ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.cylinder_index_count, 0, 0..self.cylinder_instance_count);
                }
            }

            // Shadow ligand spheres — the ligand does not *receive* shadows, but
            // it still casts them onto the protein so it feels grounded in the pocket.
            if let Some(buf) = &self.ligand_sphere_instances {
                if self.ligand_sphere_instance_count > 0 {
                    pass.set_pipeline(&self.shadow_impostor_pipeline);
                    pass.set_bind_group(0, &self.shadow_uniform_bg, &[]);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..6, 0..self.ligand_sphere_instance_count);
                }
            }

            // Shadow ligand cylinders
            if let Some(buf) = &self.ligand_cylinder_instances {
                if self.ligand_cylinder_instance_count > 0 {
                    pass.set_pipeline(&self.shadow_cylinder_pipeline);
                    pass.set_bind_group(0, &self.shadow_uniform_bg, &[]);
                    pass.set_vertex_buffer(0, self.cylinder_vb.slice(..));
                    pass.set_vertex_buffer(1, buf.slice(..));
                    pass.set_index_buffer(self.cylinder_ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.cylinder_index_count, 0, 0..self.ligand_cylinder_instance_count);
                }
            }

            // Shadow ribbon
            if let (Some(vb), Some(ib)) = (&self.ribbon_vb, &self.ribbon_ib) {
                if self.ribbon_index_count > 0 {
                    pass.set_pipeline(&self.shadow_mesh_pipeline);
                    pass.set_bind_group(0, &self.shadow_uniform_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.ribbon_index_count, 0, 0..1);
                }
            }

            // Shadow surface
            if let (Some(vb), Some(ib)) = (&self.surface_vb, &self.surface_ib) {
                if self.surface_index_count > 0 {
                    pass.set_pipeline(&self.shadow_mesh_pipeline);
                    pass.set_bind_group(0, &self.shadow_uniform_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.surface_index_count, 0, 0..1);
                }
            }
        }

        // ── Pass 1: Opaque MSAA pass (Rgba16Float) ────────────────────────────
        // Renders sphere/cylinder/ribbon → msaa_color_view (MSAA×4)
        // Resolves to scene_color_view (sample_count=1)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("OpaquePass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_color_view,
                    resolve_target: Some(&self.scene_color_view),
                    ops: wgpu::Operations {
                        // Alpha=0 so post.wgsl can detect background pixels (no geometry)
                        // by checking scene_tex.a == 0.  RGB = bg_color so surface
                        // alpha-blends correctly over the intended background color.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.bg_color.r,
                            g: self.bg_color.g,
                            b: self.bg_color.b,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Draw spheres first (impostor: 6 vertices per instance, no mesh buffer)
            if let Some(buf) = &self.sphere_instances {
                pass.set_pipeline(&self.sphere_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.shadow_bg, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..6, 0..self.sphere_instance_count);
            }

            // Draw cylinders (depth test makes bonds visible through spheres at junctions)
            if let Some(buf) = &self.cylinder_instances {
                pass.set_pipeline(&self.cylinder_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.shadow_bg, &[]);
                pass.set_vertex_buffer(0, self.cylinder_vb.slice(..));
                pass.set_vertex_buffer(1, buf.slice(..));
                pass.set_index_buffer(self.cylinder_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.cylinder_index_count, 0, 0..self.cylinder_instance_count);
            }

            // Draw ribbon
            if let (Some(vb), Some(ib)) = (&self.ribbon_vb, &self.ribbon_ib) {
                pass.set_pipeline(&self.ribbon_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.shadow_bg, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.ribbon_index_count, 0, 0..1);
            }
        }

        // ── Pass 2: Depth resolve (MSAA → single-sample) ──────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DepthResolvePass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_single_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.depth_resolve_pipeline);
            pass.set_bind_group(0, &self.depth_resolve_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 2.5: Ligand overlay ─────────────────────────────────────────
        // Drawn BEFORE the surface (against protein-only depth) so the ligand
        // sits at its true depth: a semi-transparent surface then blends over it
        // and it shows through — dimmed by the transparency, just like the ribbon —
        // while the ligand still occludes the surface where it is in front.
        if self.ligand_sphere_instances.is_some() || self.ligand_cylinder_instances.is_some() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("LigandOverlayPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_single_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if let Some(buf) = &self.ligand_sphere_instances {
                pass.set_pipeline(&self.ligand_sphere_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.shadow_bg, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..6, 0..self.ligand_sphere_instance_count);
            }

            if let Some(buf) = &self.ligand_cylinder_instances {
                pass.set_pipeline(&self.ligand_cylinder_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.shadow_bg, &[]);
                pass.set_vertex_buffer(0, self.cylinder_vb.slice(..));
                pass.set_vertex_buffer(1, buf.slice(..));
                pass.set_index_buffer(self.cylinder_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.cylinder_index_count, 0, 0..self.ligand_cylinder_instance_count);
            }
        }

        // ── Pass 3: Surface alpha-blend pass ─────────────────────────────────
        // Renders BEFORE SSAO so that depth_single_tex includes surface depth,
        // preventing opaque geometry (e.g. ligands) from leaking SSAO shadows
        // through the surface.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SurfacePass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_single_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        // Store: depth_single_tex is read by SSAO and Post Sobel.
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if let (Some(vb), Some(ib)) = (&self.surface_vb, &self.surface_ib) {
                pass.set_pipeline(&self.surface_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.shadow_bg, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.surface_index_count, 0, 0..1);
            }
        }

        // ── Pass 4: SSAO pass ───────────────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAOPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.ssao_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 4.5: SSAO blur pass ────────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAOBlurPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_blur_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.ssao_blur_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.ssao_blur_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 4.6: Bloom downsample ───────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("BloomDown"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_down_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.bloom_down_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 4.7: Bloom blur H (bloom_a → bloom_b) ──────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("BloomBlurH"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_b_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_blur_h_pipeline);
            pass.set_bind_group(0, &self.bloom_blur_h_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 4.8: Bloom blur V (bloom_b → bloom_a) ──────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("BloomBlurV"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.bloom_blur_v_pipeline);
            pass.set_bind_group(0, &self.bloom_blur_v_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 5: Post composite pass ───────────────────────────────────────
        // SSAO + Sobel edge + Bloom + ACES → output_view (sRGB)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PostPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.post_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.post_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Screenshot capture (between post-composite and egui) ─────────────
        // Re-run the post-composite pass to a separate COPY_SRC texture so we
        // capture the final image without egui UI overlay.
        let screenshot_staging = if self.pending_screenshot.is_some() {
            let w = self.config.width;
            let h = self.config.height;
            let capture_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ScreenshotCapture"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let capture_view = capture_tex.create_view(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ScreenshotPostPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &capture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.post_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.post_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            let bytes_per_pixel = 4u32;
            let unpadded_bytes_per_row = w * bytes_per_pixel;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
            let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ScreenshotStaging"),
                size: (padded_bytes_per_row * h) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &capture_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging_buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            Some((staging_buf, w, h, padded_bytes_per_row, unpadded_bytes_per_row))
        } else {
            None
        };

        // ── Pass 6: egui overlay ──────────────────────────────────────────────
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("EguiPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, egui_primitives, screen_desc);
        }

        self.queue.submit([encoder.finish()]);
        output.present();

        // Release egui textures that are no longer needed.
        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // ── Screenshot readback & save ────────────────────────────────────────
        if let (Some((staging_buf, w, h, padded_bpr, unpadded_bpr)), Some(path)) =
            (screenshot_staging, self.pending_screenshot.take())
        {
            let slice = staging_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            self.device.poll(wgpu::Maintain::Wait);
            match rx.recv() {
                Ok(Ok(())) => {
                    let data = slice.get_mapped_range();
                    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
                    for row in 0..h {
                        let offset = (row * padded_bpr) as usize;
                        let row_data = &data[offset..offset + unpadded_bpr as usize];
                        // BGRA → RGBA
                        for pixel in row_data.chunks_exact(4) {
                            rgba.push(pixel[2]); // R
                            rgba.push(pixel[1]); // G
                            rgba.push(pixel[0]); // B
                            rgba.push(pixel[3]); // A
                        }
                    }
                    drop(data);
                    staging_buf.unmap();

                    match image::RgbaImage::from_raw(w, h, rgba) {
                        Some(img) => {
                            if let Err(e) = img.save(&path) {
                                eprintln!("png: failed to save '{}': {e}", path.display());
                            } else {
                                println!("Saved screenshot: {} ({}×{})", path.display(), w, h);
                            }
                        }
                        None => eprintln!("png: failed to create image buffer"),
                    }
                }
                Ok(Err(e)) => eprintln!("png: buffer map failed: {e}"),
                Err(_) => eprintln!("png: buffer map channel closed"),
            }
        }

        Ok(())
    }
}

// ── Bind group helpers ────────────────────────────────────────────────────────

fn create_depth_resolve_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    msaa_depth_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("DepthResolveBG"),
        layout: bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(msaa_depth_view),
        }],
    })
}

fn create_ssao_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    depth_single_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SSAOBG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_single_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn create_ssao_blur_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    ssao_view: &wgpu::TextureView,
    depth_single_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SSAOBlurBG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(ssao_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_single_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

fn create_post_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    scene_view: &wgpu::TextureView,
    ssao_view: &wgpu::TextureView,
    depth_single_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    bloom_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PostBG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(ssao_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_single_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(bloom_view) },
        ],
    })
}

/// Keep only triangles in connected mesh components (via shared welded vertices)
/// whose triangle count is at least `min_ratio` of the largest component.
/// Small isolated fragments — e.g. stray back-side patches left after the pocket
/// clip — are dropped. Vertices are already welded per chain, so shared vertex
/// indices imply connectivity.
fn drop_small_components(n_verts: usize, idxs: Vec<u32>, min_ratio: f32) -> Vec<u32> {
    if idxs.is_empty() {
        return idxs;
    }

    // Union-Find over vertices.
    let mut parent: Vec<u32> = (0..n_verts as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize]; // path halving
            x = parent[x as usize];
        }
        x
    }
    for t in idxs.chunks(3) {
        for &b in &[t[1], t[2]] {
            let ra = find(&mut parent, t[0]);
            let rb = find(&mut parent, b);
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }
    }

    // Flatten to a root per vertex, then count triangles per component.
    let roots: Vec<u32> = (0..n_verts as u32).map(|v| find(&mut parent, v)).collect();
    let mut count: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for t in idxs.chunks(3) {
        *count.entry(roots[t[0] as usize]).or_default() += 1;
    }
    let max = count.values().copied().max().unwrap_or(0);
    let threshold = ((max as f32) * min_ratio).ceil() as usize;

    idxs.chunks(3)
        .filter(|t| count[&roots[t[0] as usize]] >= threshold)
        .flat_map(|t| t.iter().copied())
        .collect()
}

/// Fill holes left in a triangle mesh (e.g. by the pocket clip). Boundary edges
/// (used by a single triangle) are chained into closed loops; every loop except
/// the largest — which is the intended open rim — is triangulated with a fan,
/// as long as it has at most `max_hole_edges` edges. Fill triangles reuse the
/// loop's existing vertices, so no new vertices or normals are needed.
fn fill_mesh_holes(mut idxs: Vec<u32>, max_hole_edges: usize) -> Vec<u32> {
    use std::collections::{HashMap, HashSet};
    if idxs.len() < 3 {
        return idxs;
    }

    // All directed edges present in the mesh.
    let mut dir_edges: HashSet<(u32, u32)> = HashSet::with_capacity(idxs.len());
    for t in idxs.chunks(3) {
        dir_edges.insert((t[0], t[1]));
        dir_edges.insert((t[1], t[2]));
        dir_edges.insert((t[2], t[0]));
    }

    // Boundary directed edge (u,v): its reverse (v,u) is absent. For a manifold
    // boundary each vertex has exactly one outgoing boundary edge → a next map.
    let mut next: HashMap<u32, u32> = HashMap::new();
    for &(u, v) in &dir_edges {
        if !dir_edges.contains(&(v, u)) {
            next.insert(u, v);
        }
    }
    if next.is_empty() {
        return idxs;
    }

    // Chain boundary edges into closed loops.
    let mut visited: HashSet<u32> = HashSet::new();
    let mut loops: Vec<Vec<u32>> = Vec::new();
    let starts: Vec<u32> = next.keys().copied().collect();
    for start in starts {
        if visited.contains(&start) {
            continue;
        }
        let mut lp = Vec::new();
        let mut cur = start;
        while visited.insert(cur) {
            lp.push(cur);
            match next.get(&cur) {
                Some(&nv) => cur = nv,
                None => break,
            }
        }
        if lp.len() >= 3 {
            loops.push(lp);
        }
    }

    // The largest loop is the intended open rim; fill the rest (holes).
    let largest = loops
        .iter()
        .enumerate()
        .max_by_key(|(_, l)| l.len())
        .map(|(i, _)| i);
    for (i, lp) in loops.iter().enumerate() {
        if Some(i) == largest || lp.len() > max_hole_edges {
            continue;
        }
        for k in 1..lp.len() - 1 {
            idxs.push(lp[0]);
            idxs.push(lp[k]);
            idxs.push(lp[k + 1]);
        }
    }
    idxs
}

// ── Pipeline builder ──────────────────────────────────────────────────────────

fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    vs_entry: &str,
    fs_entry: &str,
    buffers: &[wgpu::VertexBufferLayout<'_>],
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs_entry),
            compilation_options: Default::default(),
            buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // No culling: both tube walls render; depth test keeps the near wall,
            // so thin capless cylinders always read as solid rods (not half-pipes).
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: sample_count, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None,
        cache: None,
    })
}

// ── Texture creation helpers ──────────────────────────────────────────────────

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DepthTexture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        // TEXTURE_BINDING so depth_resolve shader can read it
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_depth_single_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DepthSingle"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_msaa_color_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAAColor"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_rgba16float_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
    usage: wgpu::TextureUsages,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_r8unorm_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SSAO"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// Compute per-atom residue identifiers for a structure.
fn create_bloom_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_bloom_down_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    scene_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BloomDownBG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

fn create_bloom_blur_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    src_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

fn compute_residue_ids(structure: &crate::structure::atom::Structure) -> Vec<u32> {
    let atoms = &structure.atoms;
    let mut ids = vec![0u32; atoms.len()];
    let mut first = 0u32;
    for i in 0..atoms.len() {
        if i == 0 || {
            let a = &atoms[i];
            let p = &atoms[i - 1];
            a.residue.chain   != p.residue.chain
            || a.residue.seq_num  != p.residue.seq_num
            || a.residue.ins_code != p.residue.ins_code
        } {
            first = i as u32;
        }
        ids[i] = first;
    }
    ids
}
