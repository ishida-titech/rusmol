use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use std::collections::HashMap;

use crate::render::ball_stick::{CylinderInstance, SphereInstance, Vertex};
use crate::render::camera::Camera;
use crate::render::picker::Picker;
use crate::render::ribbon::{build_ribbon, RibbonVertex};
use crate::render::surface::build_surface;
use crate::render::uniform::Uniforms;
use crate::scene::object::{RepresentationType, REP_BACKBONE, REP_BALL_STICK, REP_RIBBON, REP_SURFACE};
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

const BOND_RADIUS: f32 = 0.15;
const BACKBONE_TUBE_RADIUS: f32 = 0.30;
const BACKBONE_JOINT_RADIUS: f32 = 0.36;

pub struct RenderState {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    // ── Sphere pipeline ──────────────────────────────────────────────────────
    sphere_pipeline: wgpu::RenderPipeline,
    sphere_vb: wgpu::Buffer,
    sphere_ib: wgpu::Buffer,
    sphere_index_count: u32,
    sphere_instances: Option<wgpu::Buffer>,
    sphere_instance_count: u32,

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

    // ── Surface pipeline ─────────────────────────────────────────────────────
    surface_pipeline: wgpu::RenderPipeline,
    surface_vb: Option<wgpu::Buffer>,
    surface_ib: Option<wgpu::Buffer>,
    surface_index_count: u32,

    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,

    // ── Phase 5: picking ─────────────────────────────────────────────────────
    picker: Picker,
    /// Maps sphere instance index (0-based) → (object_name, atom_index)
    sphere_instance_map: Vec<AtomRef>,

    /// Ghost spheres: invisible in main pass, used for Ribbon/Surface picking.
    /// Contains all non-HETATM, non-water atoms from objects with Ribbon or Surface active.
    ghost_instances: Option<wgpu::Buffer>,
    ghost_instance_count: u32,
    ghost_instance_map: Vec<AtomRef>,

    /// Per-object residue_id arrays: maps atom index → residue identifier.
    /// Built in upload_scene, used by pick_at and highlight logic.
    residue_ids_cache: HashMap<String, Vec<u32>>,

    /// Currently highlighted residue_id (0 = no highlight).
    /// Written to the GPU uniform on every update_uniforms call.
    picked_residue_id: u32,

    pub bg_color: wgpu::Color,

    /// Light intensity multiplier (default 1.0).
    pub light_intensity: f32,
    /// Light elevation angle in degrees above the horizontal (default 30.0).
    pub light_elevation_deg: f32,
    /// Light azimuth angle in degrees clockwise from forward (default 20.0).
    pub light_azimuth_deg: f32,

    /// egui overlay renderer (draws toolbar on top of the 3-D scene).
    pub egui_renderer: egui_wgpu::Renderer,
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

        let (depth_texture, depth_view) = create_depth_texture(&device, &config);

        // ── Uniform buffer ───────────────────────────────────────────────────
        let uniforms = Uniforms::new(
            glam::Mat4::IDENTITY,
            glam::Vec3::new(1.0, 1.0, 1.0),
            glam::Vec3::new(0.0, 0.0, 5.0),
            0,
            1.0,
        );
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // ── Sphere pipeline ──────────────────────────────────────────────────
        let sphere_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SphereShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/main.wgsl").into()),
        });
        let sphere_pipeline = build_pipeline(
            &device,
            &pipeline_layout,
            &sphere_shader,
            "vs_main",
            "fs_main",
            &[Vertex::desc(), SphereInstance::desc()],
            config.format,
        );
        let (s_verts, s_indices) = crate::render::ball_stick::icosphere(2);
        let sphere_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SphereVB"),
            contents: bytemuck::cast_slice(&s_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sphere_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SphereIB"),
            contents: bytemuck::cast_slice(&s_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let sphere_index_count = s_indices.len() as u32;

        // ── Cylinder pipeline ────────────────────────────────────────────────
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
            config.format,
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

        // ── Ribbon pipeline ──────────────────────────────────────────────────
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
                    format,
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
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ── Surface pipeline ─────────────────────────────────────────────────
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
                    format,
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
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ── Phase 5: picker ──────────────────────────────────────────────────
        let picker = Picker::new(&device, size.width.max(1), size.height.max(1), &bind_group_layout);

        // ── egui renderer ────────────────────────────────────────────────────
        // Renders the UI toolbar overlay on top of the 3-D scene.
        // No depth buffer needed for 2-D overlay; no MSAA.
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            uniform_buffer,
            uniform_bind_group,
            sphere_pipeline,
            sphere_vb,
            sphere_ib,
            sphere_index_count,
            sphere_instances: None,
            sphere_instance_count: 0,
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
            egui_renderer,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        let (dt, dv) = create_depth_texture(&self.device, &self.config);
        self.depth_texture = dt;
        self.depth_view = dv;
        self.picker.resize(&self.device, width, height);
    }

    /// Rebuild all geometry buffers from scene data.
    pub fn upload_scene(&mut self, scene: &Scene) {
        let _upload_t0 = std::time::Instant::now();
        let mut spheres:        Vec<SphereInstance>   = Vec::new();
        let mut sphere_map:     Vec<AtomRef>          = Vec::new();
        let mut cylinders:      Vec<CylinderInstance> = Vec::new();
        let mut ribbon_verts:   Vec<RibbonVertex>     = Vec::new();
        let mut ribbon_idxs:    Vec<u32>              = Vec::new();
        let mut surface_verts:  Vec<RibbonVertex>     = Vec::new();
        let mut surface_idxs:   Vec<u32>              = Vec::new();

        self.residue_ids_cache.clear();
        for (obj_name, obj) in scene.iter() {
            if !obj.is_visible() {
                continue;
            }
            let atoms  = &obj.structure.atoms;
            let colors = &obj.atom_colors;
            let residue_ids = compute_residue_ids(&obj.structure);
            self.residue_ids_cache.insert(obj_name.clone(), residue_ids);

            // ── Ball-and-stick ────────────────────────────────────────────────
            // Draw a sphere for each atom that has REP_BALL_STICK set.
            // Draw a bond only when both endpoints have REP_BALL_STICK set.
            for (i, atom) in atoms.iter().enumerate() {
                if obj.atom_rep_show.get(i).copied().unwrap_or(0) & REP_BALL_STICK == 0 {
                    continue;
                }
                let is_water = atom.is_hetatm
                    && matches!(atom.residue.name.as_str(), "HOH" | "WAT" | "DOD");
                let color  = colors[i];
                let radius = vdw_radius(&atom.element) * if is_water { 0.14 } else { 0.32 };
                sphere_map.push((obj_name.clone(), i));
                spheres.push(SphereInstance { position: atom.position.to_array(), radius, color, _pad: 0.0 });
            }
            for bond in &obj.structure.bonds {
                let (a1, a2) = (bond.atom1, bond.atom2);
                if a1 >= atoms.len() || a2 >= atoms.len() { continue; }
                let f1 = obj.atom_rep_show.get(a1).copied().unwrap_or(0);
                let f2 = obj.atom_rep_show.get(a2).copied().unwrap_or(0);
                if f1 & REP_BALL_STICK == 0 || f2 & REP_BALL_STICK == 0 { continue; }
                let p1  = atoms[a1].position.to_array();
                let p2  = atoms[a2].position.to_array();
                let mid = [(p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5, (p1[2]+p2[2])*0.5];
                cylinders.push(CylinderInstance::new(p1,  mid, BOND_RADIUS, colors[a1]));
                cylinders.push(CylinderInstance::new(mid, p2,  BOND_RADIUS, colors[a2]));
            }

            // ── Ribbon ───────────────────────────────────────────────────────
            if obj.has_representation(RepresentationType::Ribbon) {
                let rids = self.residue_ids_cache.get(obj_name).map(|v| v.as_slice()).unwrap_or(&[]);
                build_ribbon(&obj.structure, &obj.atom_colors, rids, &obj.atom_rep_show, &mut ribbon_verts, &mut ribbon_idxs);
            }

            // ── Surface ───────────────────────────────────────────────────────
            if obj.has_representation(RepresentationType::Surface) {
                let t0 = std::time::Instant::now();
                let rids = self.residue_ids_cache.get(obj_name).map(|v| v.as_slice()).unwrap_or(&[]);
                build_surface(&obj.structure, &obj.atom_colors, rids, &obj.atom_rep_show, &mut surface_verts, &mut surface_idxs);
                log::info!(
                    "surface build '{}': {:.0} ms  ({} verts, {} tris)",
                    obj_name,
                    t0.elapsed().as_secs_f64() * 1000.0,
                    surface_verts.len(),
                    surface_idxs.len() / 3,
                );
            }

            // ── Backbone (Cα trace) ───────────────────────────────────────────
            if obj.has_representation(RepresentationType::Backbone) {
                // Collect Cα atoms per chain, sorted by (seq_num, ins_code)
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
                    // Joint spheres at each Cα
                    for &(_, _, i) in chain_cas.iter() {
                        sphere_map.push((obj_name.clone(), i));
                        spheres.push(SphereInstance {
                            position: atoms[i].position.to_array(),
                            radius: BACKBONE_JOINT_RADIUS,
                            color: colors[i],
                            _pad: 0.0,
                        });
                    }
                    // Tube segments between consecutive Cα atoms
                    for window in chain_cas.windows(2) {
                        let (_, _, i1) = window[0];
                        let (_, _, i2) = window[1];
                        let p1  = atoms[i1].position.to_array();
                        let p2  = atoms[i2].position.to_array();
                        let mid = [(p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5, (p1[2]+p2[2])*0.5];
                        cylinders.push(CylinderInstance::new(p1,  mid, BACKBONE_TUBE_RADIUS, colors[i1]));
                        cylinders.push(CylinderInstance::new(mid, p2,  BACKBONE_TUBE_RADIUS, colors[i2]));
                    }
                }
            }
        }

        // ── Ghost spheres for Ribbon / Surface picking ────────────────────────
        // All non-HETATM non-water atoms from objects that have Ribbon or Surface active.
        // Rendered only in the pick pass (invisible in main pass).
        let mut ghost_spheres: Vec<SphereInstance> = Vec::new();
        let mut ghost_map: Vec<AtomRef> = Vec::new();
        for (obj_name, obj) in scene.iter() {
            if !obj.is_visible() {
                continue;
            }
            for (i, atom) in obj.structure.atoms.iter().enumerate() {
                let flags = obj.atom_rep_show.get(i).copied().unwrap_or(0);
                let atom_has_ribbon  = flags & REP_RIBBON  != 0;
                let atom_has_surface = flags & REP_SURFACE != 0;
                if !atom_has_ribbon && !atom_has_surface {
                    continue;
                }
                let is_water = matches!(atom.residue.name.as_str(), "HOH" | "WAT" | "DOD");
                if is_water {
                    continue;
                }
                // For ribbon-only atoms, restrict ghost spheres to backbone atoms
                // so picking matches what the ribbon visually represents.
                if atom_has_ribbon && !atom_has_surface {
                    let name = atom.name.trim();
                    if !matches!(name, "N" | "CA" | "C" | "O") {
                        continue;
                    }
                }
                ghost_map.push((obj_name.clone(), i));
                ghost_spheres.push(SphereInstance {
                    position: atom.position.to_array(),
                    radius: vdw_radius(&atom.element),
                    color: [0.0, 0.0, 0.0], // color unused in pick pass
                    _pad: 0.0,
                });
            }
        }
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

        self.surface_index_count = surface_idxs.len() as u32;
        if surface_verts.is_empty() {
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

        log::info!(
            "upload_scene: {:.0} ms  (spheres={}, cyls={}, ribbon_tris={}, surface_tris={})",
            _upload_t0.elapsed().as_secs_f64() * 1000.0,
            spheres.len(),
            cylinders.len(),
            ribbon_idxs.len() / 3,
            surface_idxs.len() / 3,
        );
    }

    pub fn update_uniforms(&self, camera: &Camera) {
        let view  = camera.view_matrix();
        let proj  = camera.projection_matrix();
        // Compute light direction from elevation/azimuth angles in camera space.
        let az = self.light_azimuth_deg.to_radians();
        let el = self.light_elevation_deg.to_radians();
        let light_base = glam::Vec3::new(
            el.cos() * az.sin(),
            el.sin(),
            el.cos() * az.cos(),
        );
        let light_dir = camera.rotation * light_base;
        let uniforms = Uniforms::new(
            proj * view,
            light_dir,
            camera.eye_position(),
            self.picked_residue_id,
            self.light_intensity,
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
    ///
    /// Phase 1 — exact pixel hit on rendered spheres (BallAndStick) → `PickResult::Atom`.
    /// Phase 2 — nearest-search on ghost spheres within `GHOST_PICK_RADIUS` pixels
    ///            (Ribbon / Surface) → `PickResult::Residue`.
    pub fn pick_at(&self, px: u32, py: u32) -> Option<PickResult> {
        // Phase 1: exact hit on render spheres (atom-level).
        if let Some(instances) = &self.sphere_instances {
            if self.sphere_instance_count > 0 {
                if let Some(idx) = self.picker.pick_at(
                    &self.device,
                    &self.queue,
                    &self.uniform_bind_group,
                    &self.sphere_vb,
                    &self.sphere_ib,
                    self.sphere_index_count,
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
                    &self.sphere_vb,
                    &self.sphere_ib,
                    self.sphere_index_count,
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
        let view = output.texture.create_view(&Default::default());

        // Upload any new egui textures.
        for (id, delta) in &textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Upload egui vertex/index buffers into the encoder.
        self.egui_renderer.update_buffers(
            &self.device, &self.queue, &mut encoder, egui_primitives, screen_desc,
        );

        // ── 3-D main pass ─────────────────────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MainPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.bg_color),
                        store: wgpu::StoreOp::Store,
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

            // Draw cylinders first (spheres will cover bond joints)
            if let Some(buf) = &self.cylinder_instances {
                pass.set_pipeline(&self.cylinder_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_vertex_buffer(0, self.cylinder_vb.slice(..));
                pass.set_vertex_buffer(1, buf.slice(..));
                pass.set_index_buffer(self.cylinder_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.cylinder_index_count, 0, 0..self.cylinder_instance_count);
            }

            // Draw ribbon
            if let (Some(vb), Some(ib)) = (&self.ribbon_vb, &self.ribbon_ib) {
                pass.set_pipeline(&self.ribbon_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.ribbon_index_count, 0, 0..1);
            }

            // Draw surface
            if let (Some(vb), Some(ib)) = (&self.surface_vb, &self.surface_ib) {
                pass.set_pipeline(&self.surface_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.surface_index_count, 0, 0..1);
            }

            // Draw spheres on top
            if let Some(buf) = &self.sphere_instances {
                pass.set_pipeline(&self.sphere_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_vertex_buffer(0, self.sphere_vb.slice(..));
                pass.set_vertex_buffer(1, buf.slice(..));
                pass.set_index_buffer(self.sphere_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.sphere_index_count, 0, 0..self.sphere_instance_count);
            }
        }

        // ── egui overlay pass ────────────────────────────────────────────────
        // LoadOp::Load preserves the 3-D scene below the UI.
        // forget_lifetime() is required by egui_wgpu's render() API.
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("EguiPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
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
}

fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    vs_entry: &str,
    fs_entry: &str,
    buffers: &[wgpu::VertexBufferLayout<'_>],
    format: wgpu::TextureFormat,
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
            cull_mode: Some(wgpu::Face::Back),
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
    })
}

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DepthTexture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// Compute per-atom residue identifiers for a structure.
/// The identifier is the index of the first atom in the same residue
/// (grouped by chain + seq_num + ins_code). All atoms in one residue
/// share the same value, enabling exact equality tests in shaders.
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
