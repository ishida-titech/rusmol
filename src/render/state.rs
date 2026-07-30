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
/// Gold/yellow highlight for auto-detected covalent protein–ligand links.
const COVALENT_BOND_COLOR: [f32; 3] = [1.0, 0.82, 0.10];
/// Covalent link bonds are drawn thicker than normal sticks.
const COVALENT_BOND_RADIUS: f32 = BOND_RADIUS * 1.3;
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

/// Bind-group layouts and the shared sampler needed to build a [`RenderTargets`].
/// The layouts and sampler live on [`RenderState`]; only the views/targets change
/// when the resolution changes.
struct TargetLayouts<'a> {
    depth_resolve: &'a wgpu::BindGroupLayout,
    ssao: &'a wgpu::BindGroupLayout,
    ssao_blur: &'a wgpu::BindGroupLayout,
    bloom_down: &'a wgpu::BindGroupLayout,
    bloom_blur: &'a wgpu::BindGroupLayout,
    post: &'a wgpu::BindGroupLayout,
    dof: &'a wgpu::BindGroupLayout,
    sampler: &'a wgpu::Sampler,
}

/// All offscreen render targets plus the bind groups that depend solely on their
/// views. Built at an arbitrary width/height so the exact same scene passes can
/// render either to the swapchain (live) or to a hi-res capture (offline export),
/// independent of the surface configuration. Texture formats are fixed
/// (Rgba16Float color, Depth32Float depth, R8Unorm SSAO); bloom is half-res.
// The owning `wgpu::Texture` handles are retained so the textures outlive their
// views for as long as the targets are in use (matching the previous per-field
// storage on `RenderState`); only the views/bind groups are read at draw time.
#[allow(dead_code)]
pub struct RenderTargets {
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

    // ── SSAO textures ─────────────────────────────────────────────────────────
    ssao_tex: wgpu::Texture,
    ssao_view: wgpu::TextureView,
    ssao_blur_tex: wgpu::Texture,
    ssao_blur_view: wgpu::TextureView,

    // ── Bloom (half-res) ─────────────────────────────────────────────────────
    bloom_a_tex: wgpu::Texture,
    bloom_a_view: wgpu::TextureView,
    bloom_b_tex: wgpu::Texture,
    bloom_b_view: wgpu::TextureView,

    // ── Depth-of-field (full res; copied back over scene_color) ───────────────
    dof_tex: wgpu::Texture,
    dof_view: wgpu::TextureView,

    // ── Bind groups depending solely on the views above ──────────────────────
    depth_resolve_bg: wgpu::BindGroup,
    ssao_bg: wgpu::BindGroup,
    ssao_blur_bg: wgpu::BindGroup,
    bloom_down_bg: wgpu::BindGroup,
    bloom_blur_h_bg: wgpu::BindGroup,   // reads bloom_a, writes bloom_b
    bloom_blur_v_bg: wgpu::BindGroup,   // reads bloom_b, writes bloom_a
    post_bg: wgpu::BindGroup,
    dof_bg: wgpu::BindGroup,            // reads scene_color + depth_single
}

impl RenderTargets {
    /// Allocate every offscreen target and its dependent bind groups at
    /// `width`×`height`. Does not touch the surface/swapchain.
    fn new(device: &wgpu::Device, width: u32, height: u32, l: &TargetLayouts) -> Self {
        let width = width.max(1);
        let height = height.max(1);

        let (depth_texture, depth_view) = create_depth_texture(device, width, height, 4);
        let (msaa_texture, msaa_color_view) = create_msaa_color_texture(device, width, height);
        let (scene_color_tex, scene_color_view) = create_rgba16float_texture(
            device, width, height, 1,
            // COPY_DST so the DoF pass result (dof_tex) can be copied back over it.
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            "SceneColor",
        );
        let (depth_single_tex, depth_single_view) = create_depth_single_texture(device, width, height);
        let (ssao_tex, ssao_view) = create_r8unorm_texture(device, width, height);
        let (ssao_blur_tex, ssao_blur_view) = create_r8unorm_texture(device, width, height);

        // Bloom is half-res.
        let bloom_half_w = (width / 2).max(1);
        let bloom_half_h = (height / 2).max(1);
        let (bloom_a_tex, bloom_a_view) = create_bloom_texture(device, bloom_half_w, bloom_half_h, "BloomA");
        let (bloom_b_tex, bloom_b_view) = create_bloom_texture(device, bloom_half_w, bloom_half_h, "BloomB");

        // DoF target: full-res HDR, copyable back onto scene_color.
        let (dof_tex, dof_view) = create_rgba16float_texture(
            device, width, height, 1,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            "DoF",
        );

        let depth_resolve_bg = create_depth_resolve_bg(device, l.depth_resolve, &depth_view);
        let ssao_bg = create_ssao_bg(device, l.ssao, &depth_single_view, l.sampler);
        let ssao_blur_bg = create_ssao_blur_bg(
            device, l.ssao_blur, &ssao_view, &depth_single_view, l.sampler,
        );
        let bloom_down_bg = create_bloom_down_bg(device, l.bloom_down, &scene_color_view, l.sampler);
        let bloom_blur_h_bg = create_bloom_blur_bg(device, l.bloom_blur, &bloom_a_view, l.sampler, "BloomBlurH_BG");
        let bloom_blur_v_bg = create_bloom_blur_bg(device, l.bloom_blur, &bloom_b_view, l.sampler, "BloomBlurV_BG");
        let post_bg = create_post_bg(
            device, l.post,
            &scene_color_view, &ssao_blur_view, &depth_single_view, l.sampler,
            &bloom_a_view,
        );
        let dof_bg = create_dof_bg(device, l.dof, &scene_color_view, &depth_single_view, l.sampler);

        Self {
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
            ssao_blur_tex,
            ssao_blur_view,
            bloom_a_tex,
            bloom_a_view,
            bloom_b_tex,
            bloom_b_view,
            dof_tex,
            dof_view,
            depth_resolve_bg,
            ssao_bg,
            ssao_blur_bg,
            bloom_down_bg,
            bloom_blur_h_bg,
            bloom_blur_v_bg,
            post_bg,
            dof_bg,
        }
    }
}

pub struct RenderState {
    /// The swapchain surface. `None` in headless mode (no window); rendering to
    /// the swapchain is then a no-op and only offscreen `export` is used.
    pub surface: Option<wgpu::Surface<'static>>,
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

    // ── Offscreen render targets + their bind groups (rebuilt on resize; a
    //    temporary hi-res instance is built for offline export) ────────────────
    targets: RenderTargets,

    // ── Post-process pipelines ────────────────────────────────────────────────
    depth_resolve_pipeline: wgpu::RenderPipeline,
    ssao_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,
    ssao_blur_pipeline: wgpu::RenderPipeline,
    dof_pipeline: wgpu::RenderPipeline,

    // ── Bind group layouts (the bind groups themselves live in `targets`) ─────
    depth_resolve_bgl: wgpu::BindGroupLayout,
    ssao_bgl: wgpu::BindGroupLayout,
    ssao_blur_bgl: wgpu::BindGroupLayout,
    post_bgl: wgpu::BindGroupLayout,
    dof_bgl: wgpu::BindGroupLayout,

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
    /// When true, subtract bound-ligand volume from the protein surface so a
    /// covalent/bound ligand sits in a carved cavity instead of being engulfed.
    /// Toggled via `set surface_carve_ligand, 0|1` (default off).
    pub surface_carve_ligand: bool,
    /// When true (default), auto-detect covalent protein–ligand links (from
    /// CONECT/topology) and highlight them in gold, showing the partner residue
    /// as sticks. Toggled via `set show_covalent, 0|1`.
    pub show_covalent: bool,
    /// Ligand–protein hydrogen bonds (heavy-atom endpoint pairs) drawn as dashed
    /// lines in the Binding Site view. Empty in all other presets.
    pub hbond_segments: Vec<(glam::Vec3, glam::Vec3)>,

    // ── Shadow mapping ───────────────────────────────────────────────────────
    shadow_map_view: wgpu::TextureView,
    shadow_uniform_buffer: wgpu::Buffer,
    shadow_uniform_bg: wgpu::BindGroup,
    shadow_bg: wgpu::BindGroup,            // group 1 for main shaders
    /// Layout + sampler retained so `export` can rebuild `shadow_bg` at a
    /// temporarily higher shadow-map resolution.
    shadow_bgl: wgpu::BindGroupLayout,
    shadow_sampler: wgpu::Sampler,
    /// Live shadow-map resolution (default [`SHADOW_MAP_SIZE`]). `export` renders
    /// with a temporary 4096² map for sharper shadows, then restores this.
    pub shadow_map_size: u32,
    shadow_impostor_pipeline: wgpu::RenderPipeline,
    shadow_cylinder_pipeline: wgpu::RenderPipeline,
    shadow_mesh_pipeline: wgpu::RenderPipeline,
    scene_center: glam::Vec3,
    scene_radius: f32,

    // ── Bloom (bind groups + half-res textures live in `targets`) ────────────
    bloom_down_pipeline: wgpu::RenderPipeline,
    bloom_blur_h_pipeline: wgpu::RenderPipeline,
    bloom_blur_v_pipeline: wgpu::RenderPipeline,
    bloom_down_bgl: wgpu::BindGroupLayout,
    bloom_blur_bgl: wgpu::BindGroupLayout,

    /// egui overlay renderer.
    pub egui_renderer: egui_wgpu::Renderer,

    /// Supersampling factor for offline `render` export (1..=4). CPU-side only
    /// (not a shader uniform): the scene is rendered at `antialias`× resolution
    /// and box-downsampled. Default 2. Set via `set antialias, <1-4>`.
    pub antialias: u32,

    /// Transparent background for exported PNGs (default false). Set via
    /// `set transparent_bg, 0|1`. On-screen the swapchain ignores alpha.
    pub bg_transparent: bool,
    /// SSAO sample count (clamped 8..=64, default 16 = the previous hardcoded
    /// value). Set via `set ssao_samples, N`.
    pub ssao_samples: u32,
    /// Depth-of-field strength (0 = off, default 0). Set via `set dof, <0-1>`.
    pub dof_strength: f32,
    /// Depth-of-field blur scale (default 1.0). Set via `set dof_aperture, <f>`.
    pub dof_aperture: f32,
    /// Depth-of-field focus distance in Å from the camera; 0 = auto (distance to
    /// the scene center). Set via `set dof_focus, <f>`.
    pub dof_focus: f32,
}

impl RenderState {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            // PRIMARY covers Metal (macOS), Vulkan (Linux), and DX12 (Windows);
            // the platform's native backend is chosen automatically.
            backends: wgpu::Backends::PRIMARY,
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

        Ok(Self::build(device, queue, config, Some(surface)))
    }

    /// Create a headless (no-window) render state. Uses the same offscreen
    /// pipelines as [`new`]; there is no swapchain surface, so only `export`
    /// produces output. `width`/`height` set the default capture size (via the
    /// config); an actual surface is never created, so this works with no
    /// display server.
    pub async fn new_headless(width: u32, height: u32) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
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

        // No real surface: this config only carries the capture size + format.
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };

        Ok(Self::build(device, queue, config, None))
    }

    /// Shared construction from a ready device/queue/config. Builds all
    /// pipelines, offscreen targets, shadow map, and the picker. `surface` is
    /// `Some` in windowed mode and `None` in headless mode. The color format for
    /// the final composite/egui pass comes from `config.format`.
    fn build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
        surface: Option<wgpu::Surface<'static>>,
    ) -> Self {
        // Offscreen render targets are allocated below, once the bind group
        // layouts and shared sampler they depend on exist (see `targets`).

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
        let (_shadow_map_tex, shadow_map_view) =
            create_shadow_map_texture(&device, SHADOW_MAP_SIZE);
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
        let shadow_bg = create_shadow_bg(&device, &shadow_bgl, &shadow_map_view, &shadow_sampler);

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
            0,     // bg_transparent
            16,    // ssao_samples
            0.0,   // dof_strength (off)
            0.0,   // dof_focus (auto)
            1.0,   // dof_aperture
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

        // ── SSAO blur pipeline ────────────────────────────────────────────────
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
        // ── Bloom pipelines ─────────────────────────────────────────────────
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
                    format: config.format,
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

        // ── Depth-of-field pipeline ───────────────────────────────────────────
        let dof_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoFBGL"),
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
                // depth_tex (binding 1)
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
                // sampler (binding 2)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let dof_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoFPipelineLayout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &dof_bgl],
            push_constant_ranges: &[],
        });
        let dof_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoFShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dof.wgsl").into()),
        });
        let dof_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoFPipeline"),
            layout: Some(&dof_layout),
            vertex: wgpu::VertexState {
                module: &dof_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &dof_shader,
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

        // ── Offscreen render targets (all layouts + sampler now exist) ────────
        let targets = RenderTargets::new(
            &device,
            config.width,
            config.height,
            &TargetLayouts {
                depth_resolve: &depth_resolve_bgl,
                ssao: &ssao_bgl,
                ssao_blur: &ssao_blur_bgl,
                bloom_down: &bloom_down_bgl,
                bloom_blur: &bloom_blur_bgl,
                post: &post_bgl,
                dof: &dof_bgl,
                sampler: &linear_sampler,
            },
        );

        // ── Phase 5: picker ──────────────────────────────────────────────────
        let picker = Picker::new(&device, config.width, config.height, &uniform_bind_group_layout);

        // ── egui renderer ────────────────────────────────────────────────────
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1, false);

        Self {
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
            targets,
            depth_resolve_pipeline,
            ssao_pipeline,
            post_pipeline,
            ssao_blur_pipeline,
            dof_pipeline,
            depth_resolve_bgl,
            ssao_bgl,
            ssao_blur_bgl,
            post_bgl,
            dof_bgl,
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
            surface_carve_ligand: false,
            show_covalent: true,
            hbond_segments: Vec::new(),
            shadow_map_view,
            shadow_uniform_buffer,
            shadow_uniform_bg,
            shadow_bg,
            shadow_bgl,
            shadow_sampler,
            shadow_map_size: SHADOW_MAP_SIZE,
            shadow_impostor_pipeline,
            shadow_cylinder_pipeline,
            shadow_mesh_pipeline,
            scene_center: glam::Vec3::ZERO,
            scene_radius: 50.0,
            bloom_down_pipeline,
            bloom_blur_h_pipeline,
            bloom_blur_v_pipeline,
            bloom_down_bgl,
            bloom_blur_bgl,
            egui_renderer,
            antialias: 2,
            bg_transparent: false,
            ssao_samples: 16,
            dof_strength: 0.0,
            dof_aperture: 1.0,
            dof_focus: 0.0,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        // resize is only called in windowed mode; guard for the headless case.
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.config);
        }

        // Rebuild the whole offscreen target set at the new resolution. The
        // bind group layouts and shared sampler are unchanged, so only the
        // textures/views and their bind groups are recreated here.
        self.targets = RenderTargets::new(&self.device, width, height, &self.target_layouts());

        self.picker.resize(&self.device, width, height);
    }

    /// Borrow the bind-group layouts and sampler needed to build a
    /// [`RenderTargets`]. They live on `self` and never change with resolution.
    fn target_layouts(&self) -> TargetLayouts<'_> {
        TargetLayouts {
            depth_resolve: &self.depth_resolve_bgl,
            ssao: &self.ssao_bgl,
            ssao_blur: &self.ssao_blur_bgl,
            bloom_down: &self.bloom_down_bgl,
            bloom_blur: &self.bloom_blur_bgl,
            post: &self.post_bgl,
            dof: &self.dof_bgl,
            sampler: &self.linear_sampler,
        }
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

                // ── Covalent protein–ligand links (CONECT/topology) ───────────
                // A bond where exactly one endpoint is a ligand atom (HETATM,
                // non-water) and the other is a non-hetatm polymer atom, in a
                // *different* residue, is a genuine covalent link (not a distance
                // guess). We highlight the crossing bond in gold and show the
                // partner protein residue as sticks. `covalent_partner_atoms`
                // holds every atom index of those partner residues; it is used
                // to compute a transient *effective* rep (stored flags OR
                // REP_STICK) without mutating `obj.atom_rep_show`.
                let is_ligand_atom = |i: usize| -> bool {
                    let a = &atoms[i];
                    a.is_hetatm && !matches!(a.residue.name.as_str(), "HOH" | "WAT" | "DOD")
                };
                let covalent_partner_atoms: std::collections::HashSet<usize> = if self.show_covalent {
                    let mask = REP_BALL_STICK | REP_STICK;
                    let mut keys: std::collections::HashSet<(char, i32, Option<char>)> =
                        std::collections::HashSet::new();
                    for bond in &obj.structure.bonds {
                        let (a1, a2) = (bond.atom1, bond.atom2);
                        if a1 >= atoms.len() || a2 >= atoms.len() { continue; }
                        let (lig, poly) = if is_ligand_atom(a1) && !atoms[a2].is_hetatm {
                            (a1, a2)
                        } else if is_ligand_atom(a2) && !atoms[a1].is_hetatm {
                            (a2, a1)
                        } else {
                            continue;
                        };
                        let rl = &atoms[lig].residue;
                        let rp = &atoms[poly].residue;
                        // Same-residue ATOM↔HETATM is handled by the normal path.
                        if rl.chain == rp.chain && rl.seq_num == rp.seq_num && rl.ins_code == rp.ins_code {
                            continue;
                        }
                        // Only when the ligand endpoint is actually shown.
                        if obj.atom_rep_show.get(lig).copied().unwrap_or(0) & mask == 0 { continue; }
                        keys.insert((rp.chain, rp.seq_num, rp.ins_code));
                    }
                    if keys.is_empty() {
                        std::collections::HashSet::new()
                    } else {
                        atoms.iter().enumerate()
                            .filter(|(_, a)| keys.contains(&(a.residue.chain, a.residue.seq_num, a.residue.ins_code)))
                            .map(|(i, _)| i)
                            .collect()
                    }
                } else {
                    std::collections::HashSet::new()
                };

                // ── Ball-and-stick ────────────────────────────────────────────
                for (i, atom) in atoms.iter().enumerate() {
                    let flags = obj.atom_rep_show.get(i).copied().unwrap_or(0);
                    if flags & (REP_BALL_STICK | REP_STICK) == 0 {
                        // Covalent partner residue atoms are shown as sticks even
                        // without a stored stick rep. Gate on "no stored stick
                        // bit" so we never emit an atom's sphere twice.
                        if covalent_partner_atoms.contains(&i) {
                            let inst = SphereInstance {
                                position: atom.position.to_array(),
                                radius: STICK_RADIUS,
                                color: colors[i],
                                edge_boost: 0.0,
                            };
                            sphere_map.push((obj_name.clone(), i));
                            spheres.push(inst);
                        }
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
                    let mask = REP_BALL_STICK | REP_STICK;
                    // Effective rep = stored flags OR REP_STICK for covalent
                    // partner residue atoms (transient; atom_rep_show untouched).
                    let mut f1 = obj.atom_rep_show.get(a1).copied().unwrap_or(0);
                    let mut f2 = obj.atom_rep_show.get(a2).copied().unwrap_or(0);
                    if covalent_partner_atoms.contains(&a1) { f1 |= REP_STICK; }
                    if covalent_partner_atoms.contains(&a2) { f2 |= REP_STICK; }
                    if f1 & mask == 0 || f2 & mask == 0 { continue; }
                    let mut is_covalent_link = false;
                    if atoms[a1].is_hetatm != atoms[a2].is_hetatm {
                        let same_residue = atoms[a1].residue.chain == atoms[a2].residue.chain
                            && atoms[a1].residue.seq_num == atoms[a2].residue.seq_num
                            && atoms[a1].residue.ins_code == atoms[a2].residue.ins_code;
                        if !same_residue {
                            // Different-residue ATOM↔HETATM bonds are only drawn
                            // when they are genuine covalent protein–ligand links
                            // and highlighting is enabled — otherwise skip (old
                            // behavior, byte-identical when show_covalent=false).
                            let is_link = (is_ligand_atom(a1) && !atoms[a2].is_hetatm)
                                || (is_ligand_atom(a2) && !atoms[a1].is_hetatm);
                            if !(self.show_covalent && is_link) { continue; }
                            is_covalent_link = true;
                        }
                    }
                    let p1  = atoms[a1].position;
                    let p2  = atoms[a2].position;
                    let mid = (p1 + p2) * 0.5;
                    let is_ligand_a1 = atoms[a1].is_hetatm && !matches!(atoms[a1].residue.name.as_str(), "HOH" | "WAT" | "DOD");
                    let is_ligand_a2 = atoms[a2].is_hetatm && !matches!(atoms[a2].residue.name.as_str(), "HOH" | "WAT" | "DOD");
                    let eb1 = if is_ligand_a1 { 1.0 } else { 0.0 };
                    let eb2 = if is_ligand_a2 { 1.0 } else { 0.0 };
                    // Covalent protein–ligand link: one slim gold cylinder.
                    // Inset the ligand end to the atom's ball surface so the ligand
                    // atom stays fully visible instead of being swallowed, and keep
                    // it slim so it reads as a highlight rather than a fat barrel.
                    if is_covalent_link {
                        let (p_lig, p_pro, lig_idx) = if is_ligand_a1 {
                            (p1, p2, a1)
                        } else {
                            (p2, p1, a2)
                        };
                        let dir = (p_pro - p_lig).normalize_or_zero();
                        let ball = (vdw_radius(&atoms[lig_idx].element) * 0.22)
                            .min((p_pro - p_lig).length() * 0.5);
                        let start = p_lig + dir * ball;
                        ligand_cylinders.push(CylinderInstance::new(
                            start.to_array(), p_pro.to_array(),
                            COVALENT_BOND_RADIUS, COVALENT_BOND_COLOR, 1.0,
                        ));
                        continue;
                    }
                    // Stick bonds (both atoms stick-only, no ball) are thicker.
                    let stick_bond = (f1 & REP_BALL_STICK == 0) && (f2 & REP_BALL_STICK == 0);
                    let r = if stick_bond { STICK_RADIUS } else { BOND_RADIUS };
                    let c1 = CylinderInstance::new(p1.to_array(), mid.to_array(), r, colors[a1], eb1);
                    let c2 = CylinderInstance::new(mid.to_array(), p2.to_array(), r, colors[a2], eb2);
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
                    build_surface(&obj.structure, &obj.atom_colors, rids, &obj.atom_rep_show, self.surface_type, self.surface_quality, self.surface_smooth as usize, self.surface_carve_ligand, &mut surface_verts, &mut surface_idxs);
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
        let screen_size = [self.config.width as f32, self.config.height as f32];
        self.write_uniforms(camera, screen_size);
    }

    /// Write the per-frame uniform buffer using an explicit `screen_size`. The
    /// live path passes the swapchain size; offline export passes the (super-
    /// sampled) capture size so screen-space effects sample at the right scale.
    /// The projection aspect comes from `camera` (callers override its viewport
    /// for non-square exports).
    fn write_uniforms(&self, camera: &Camera, screen_size: [f32; 2]) {
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

        // Resolve auto depth-of-field focus (0 → distance from camera to scene center).
        let dof_focus = if self.dof_focus > 0.0 {
            self.dof_focus
        } else {
            (camera.eye_position() - self.scene_center).length()
        };

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
            self.bg_transparent as u32,
            self.ssao_samples.clamp(8, 64),
            self.dof_strength,
            dof_focus,
            self.dof_aperture,
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
    /// Record the full scene render (shadow map + opaque + overlays + surface +
    /// SSAO + bloom + post-composite) into `encoder`, drawing offscreen work
    /// into `targets` and writing the final tone-mapped result into `final_view`.
    ///
    /// This is the single source of truth for the scene passes, shared by the
    /// live [`render`](Self::render) path (targets = `self.targets`, final view =
    /// the swapchain) and the offline [`export`](Self::export) path (a temporary
    /// hi-res `targets` and capture texture). It does NOT write the uniform
    /// buffer (callers set `screen_size` first), touch the swapchain, draw egui,
    /// or present. The shadow map is the shared fixed-size `self.shadow_map_view`.
    fn record_scene(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &RenderTargets,
        final_view: &wgpu::TextureView,
    ) {
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
                    view: &targets.msaa_color_view,
                    resolve_target: Some(&targets.scene_color_view),
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
                    view: &targets.depth_view,
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
                    view: &targets.depth_single_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.depth_resolve_pipeline);
            pass.set_bind_group(0, &targets.depth_resolve_bg, &[]);
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
                    view: &targets.scene_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth_single_view,
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
                    view: &targets.scene_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth_single_view,
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

        // ── Pass 3.5: Depth of field ─────────────────────────────────────────
        // Only when enabled; otherwise the encoder is byte-identical to before.
        // Blurs scene_color into dof_tex (using resolved depth) then copies the
        // result back over scene_color so all downstream passes see the DoF'd
        // color unchanged.
        if self.dof_strength > 0.0 {
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DoFPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &targets.dof_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.dof_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &targets.dof_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &targets.dof_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &targets.scene_color_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                targets.dof_tex.size(),
            );
        }

        // ── Pass 4: SSAO pass ───────────────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAOPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.ssao_view,
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
            pass.set_bind_group(1, &targets.ssao_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 4.5: SSAO blur pass ────────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAOBlurPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.ssao_blur_view,
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
            pass.set_bind_group(1, &targets.ssao_blur_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 4.6: Bloom downsample ───────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("BloomDown"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.bloom_a_view,
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
            pass.set_bind_group(1, &targets.bloom_down_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 4.7: Bloom blur H (bloom_a → bloom_b) ──────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("BloomBlurH"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.bloom_b_view,
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
            pass.set_bind_group(0, &targets.bloom_blur_h_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 4.8: Bloom blur V (bloom_b → bloom_a) ──────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("BloomBlurV"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.bloom_a_view,
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
            pass.set_bind_group(0, &targets.bloom_blur_v_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 5: Post composite pass ───────────────────────────────────────
        // SSAO + Sobel edge + Bloom + ACES → final_view (sRGB)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PostPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: final_view,
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
            pass.set_bind_group(1, &targets.post_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    pub fn render(
        &mut self,
        egui_primitives: &[egui::ClippedPrimitive],
        screen_desc: &egui_wgpu::ScreenDescriptor,
        textures_delta: egui::TexturesDelta,
    ) -> anyhow::Result<()> {
        // No swapchain in headless mode: rendering to screen is a no-op.
        let Some(surface) = self.surface.as_ref() else { return Ok(()); };
        let output = match surface.get_current_texture() {
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

        // ── Scene passes 0-5 → offscreen targets, composite into the
        //    swapchain view. Shared with the offline export path.
        self.record_scene(&mut encoder, &self.targets, &output_view);

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

        Ok(())
    }

    /// Offline high-resolution export. Renders the current scene to `out_w`×`out_h`
    /// at `self.antialias`× supersampling and saves it as a PNG, entirely
    /// independent of the swapchain (the on-screen surface is never touched).
    ///
    /// Pipeline: build temporary hi-res [`RenderTargets`] + a capture texture,
    /// record the scene into them via [`record_scene`](Self::record_scene), read
    /// the capture back with the standard 256-byte row alignment, box-downsample
    /// `ss`×`ss` in **linear** space (the capture format is sRGB), and write RGBA.
    /// The live uniforms are restored before returning so the next on-screen
    /// frame is correct. Non-square exports use projection aspect `out_w/out_h`.
    pub fn export(
        &mut self,
        camera: &mut Camera,
        out_w: u32,
        out_h: u32,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        let out_w = out_w.max(1);
        let out_h = out_h.max(1);

        // Memory cap on the supersampled render resolution: reduce ss until the
        // pixel count is under the cap (matching the grid-size safety approach).
        const MAX_PIXELS: u64 = 64_000_000;
        let mut ss = self.antialias.clamp(1, 4);
        while ss > 1
            && (out_w as u64 * ss as u64) * (out_h as u64 * ss as u64) > MAX_PIXELS
        {
            ss -= 1;
            log::warn!(
                "export: supersample factor reduced to {ss}× to keep {out_w}×{out_h} render under the {MAX_PIXELS}-pixel cap"
            );
        }
        let rw = out_w * ss;
        let rh = out_h * ss;
        if (rw as u64) * (rh as u64) > MAX_PIXELS {
            log::warn!(
                "export: {rw}×{rh} still exceeds the {MAX_PIXELS}-pixel cap at 1× supersampling; proceeding anyway"
            );
        }

        // Temporary hi-res offscreen targets + capture texture (surface untouched).
        let targets = RenderTargets::new(&self.device, rw, rh, &self.target_layouts());
        let capture_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ExportCapture"),
            size: wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let capture_view = capture_tex.create_view(&Default::default());

        // Temporarily bump shadow-map resolution and SSAO quality for the export
        // frame only; the live window resources are swapped back before returning.
        const EXPORT_SHADOW_SIZE: u32 = 4096;
        let (hi_shadow_tex, hi_shadow_view) =
            create_shadow_map_texture(&self.device, EXPORT_SHADOW_SIZE);
        let hi_shadow_bg =
            create_shadow_bg(&self.device, &self.shadow_bgl, &hi_shadow_view, &self.shadow_sampler);
        let saved_shadow_view = std::mem::replace(&mut self.shadow_map_view, hi_shadow_view);
        let saved_shadow_bg = std::mem::replace(&mut self.shadow_bg, hi_shadow_bg);
        let saved_shadow_size = self.shadow_map_size;
        self.shadow_map_size = EXPORT_SHADOW_SIZE;
        let saved_ssao = self.ssao_samples;
        self.ssao_samples = self.ssao_samples.max(32);

        // Write uniforms at the supersampled resolution, with projection aspect
        // set to out_w/out_h (the ratio is preserved by the ×ss factor).
        let saved_viewport = camera.viewport;
        camera.viewport = glam::Vec2::new(out_w as f32, out_h as f32);
        self.write_uniforms(camera, [rw as f32, rh as f32]);
        camera.viewport = saved_viewport;

        // Record the full scene (shadow → post) into the hi-res capture texture.
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.record_scene(&mut encoder, &targets, &capture_view);

        // Copy capture → staging buffer with 256-byte row alignment.
        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = rw * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ExportStaging"),
            size: (padded_bytes_per_row * rh) as u64,
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
            wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);

        // Restore the live-resolution shadow resources and SSAO quality. The
        // hi-res shadow texture stays alive (local `hi_shadow_tex`) until the
        // submitted commands referencing it have been consumed.
        self.shadow_map_view = saved_shadow_view;
        self.shadow_bg = saved_shadow_bg;
        self.shadow_map_size = saved_shadow_size;
        self.ssao_samples = saved_ssao;
        drop(hi_shadow_tex);

        // Restore live uniforms so the next on-screen frame renders correctly.
        self.update_uniforms(camera);

        // Read the capture back.
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| anyhow::anyhow!("export: buffer map channel closed"))?
            .map_err(|e| anyhow::anyhow!("export: buffer map failed: {e}"))?;
        let data = slice.get_mapped_range();

        // Swapchain format is BGRA-ordered on Metal, RGBA on many Vulkan adapters.
        let swap_rb = matches!(
            self.config.format,
            wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm
        );

        // Unpad rows into a tightly-packed rw×rh RGBA buffer (still sRGB 8-bit).
        let mut hires = vec![0u8; (rw as usize) * (rh as usize) * 4];
        for row in 0..rh {
            let src = (row * padded_bytes_per_row) as usize;
            let dst = (row * rw * 4) as usize;
            let row_data = &data[src..src + unpadded_bytes_per_row as usize];
            for (i, pixel) in row_data.chunks_exact(4).enumerate() {
                let o = dst + i * 4;
                if swap_rb {
                    hires[o] = pixel[2];
                    hires[o + 1] = pixel[1];
                    hires[o + 2] = pixel[0];
                    hires[o + 3] = pixel[3];
                } else {
                    hires[o..o + 4].copy_from_slice(pixel);
                }
            }
        }
        drop(data);
        staging_buf.unmap();

        // Box-average ss×ss blocks in linear space, then re-encode to sRGB.
        let out = if ss == 1 {
            hires
        } else {
            downsample_srgb(&hires, rw, out_w, out_h, ss, self.bg_transparent)
        };

        let img = image::RgbaImage::from_raw(out_w, out_h, out)
            .ok_or_else(|| anyhow::anyhow!("export: failed to build image buffer"))?;
        img.save(path)?;
        println!("Rendered: {} ({}×{}, {}× supersampled)", path.display(), out_w, out_h, ss);
        Ok(())
    }
}

// ── sRGB downsampling helpers ───────────────────────────────────────────────────

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Box-downsample an `rw`-wide RGBA8 (sRGB-encoded) buffer by `ss`×`ss` into an
/// `out_w`×`out_h` buffer. RGB is averaged in linear light (correct for sRGB
/// textures).
///
/// When `transparent`, color is accumulated **premultiplied** (linear RGB × α)
/// and normalized by the alpha sum, so partially-covered silhouette pixels do
/// not pull the (transparent) background color into the edge — this removes the
/// dark/bright fringing you would get from a straight average. When not
/// transparent, this is the exact straight-average behavior as before (alpha is
/// averaged directly), preserving pixel-identical opaque output.
fn downsample_srgb(src: &[u8], rw: u32, out_w: u32, out_h: u32, ss: u32, transparent: bool) -> Vec<u8> {
    let mut out = vec![0u8; (out_w as usize) * (out_h as usize) * 4];
    let n = (ss * ss) as f32;
    for oy in 0..out_h {
        for ox in 0..out_w {
            let (mut r, mut g, mut b, mut a) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for dy in 0..ss {
                for dx in 0..ss {
                    let sx = ox * ss + dx;
                    let sy = oy * ss + dy;
                    let i = ((sy * rw + sx) * 4) as usize;
                    let lr = srgb_to_linear(src[i] as f32 / 255.0);
                    let lg = srgb_to_linear(src[i + 1] as f32 / 255.0);
                    let lb = srgb_to_linear(src[i + 2] as f32 / 255.0);
                    let la = src[i + 3] as f32 / 255.0;
                    if transparent {
                        r += lr * la;
                        g += lg * la;
                        b += lb * la;
                    } else {
                        r += lr;
                        g += lg;
                        b += lb;
                    }
                    a += la;
                }
            }
            let o = ((oy * out_w + ox) * 4) as usize;
            let (rl, gl, bl) = if transparent {
                if a > 0.0 { (r / a, g / a, b / a) } else { (0.0, 0.0, 0.0) }
            } else {
                (r / n, g / n, b / n)
            };
            out[o]     = (linear_to_srgb(rl) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            out[o + 1] = (linear_to_srgb(gl) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            out[o + 2] = (linear_to_srgb(bl) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            out[o + 3] = (a / n * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    out
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

fn create_dof_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    scene_view: &wgpu::TextureView,
    depth_single_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("DoFBG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_single_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

/// Create a square shadow-map depth texture (+ view) at the given resolution.
fn create_shadow_map_texture(device: &wgpu::Device, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ShadowMapTex"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

/// Build the group-1 shadow bind group (comparison-sampled depth) for the main
/// shaders. Reused by `export` when rebuilding at a higher shadow resolution.
fn create_shadow_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    shadow_view: &wgpu::TextureView,
    shadow_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ShadowBG"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(shadow_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(shadow_sampler) },
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
    width: u32,
    height: u32,
    sample_count: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DepthTexture"),
        size: wgpu::Extent3d {
            width,
            height,
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
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DepthSingle"),
        size: wgpu::Extent3d {
            width,
            height,
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
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAAColor"),
        size: wgpu::Extent3d {
            width,
            height,
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
    width: u32,
    height: u32,
    sample_count: u32,
    usage: wgpu::TextureUsages,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
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
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SSAO"),
        size: wgpu::Extent3d {
            width,
            height,
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
