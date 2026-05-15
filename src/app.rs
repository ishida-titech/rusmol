use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};
use glam::Vec2;
use crossbeam_channel::{Receiver, Sender};

use crate::command::{executor, Command, CommandResponse};
use crate::render::{camera::Camera, state::{PickResult, RenderState}};
use wgpu;
use crate::scene::Scene;
use egui_wgpu::ScreenDescriptor;

enum AppState {
    Initializing,
    Running {
        window: Arc<Window>,
        render: RenderState,
        camera: Camera,
    },
}

pub struct App {
    state: AppState,
    scene: Scene,

    // Command channels (None if no prompt thread)
    cmd_rx:   Option<Receiver<Command>>,
    resp_tx:  Option<Sender<CommandResponse>>,

    // Initial commands from -c flag
    initial_commands: Vec<String>,

    // Mouse tracking
    last_mouse_pos:  Option<Vec2>,
    left_pressed:    bool,
    right_pressed:   bool,
    /// Physical pixel position where left button was pressed (for click vs drag)
    mouse_press_pos: Option<Vec2>,

    // Re-upload GPU data when true
    scene_dirty: bool,

    // egui UI
    egui_ctx:   egui::Context,
    egui_winit: Option<egui_winit::State>,
}

impl App {
    pub fn new(
        scene: Scene,
        cmd_rx: Option<Receiver<Command>>,
        resp_tx: Option<Sender<CommandResponse>>,
        initial_commands: Vec<String>,
    ) -> Self {
        Self {
            state: AppState::Initializing,
            scene,
            cmd_rx,
            resp_tx,
            initial_commands,
            last_mouse_pos: None,
            left_pressed: false,
            right_pressed: false,
            mouse_press_pos: None,
            scene_dirty: false,
            egui_ctx:   egui::Context::default(),
            egui_winit: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let t_start = std::time::Instant::now();
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("rusmol")
                        .with_inner_size(PhysicalSize::new(1600u32, 1600u32)),
                )
                .expect("Failed to create window"),
        );
        log::info!("window created: {:.0} ms", t_start.elapsed().as_secs_f64() * 1000.0);

        let mut render = pollster::block_on(RenderState::new(window.clone()))
            .expect("Failed to initialize GPU");
        log::info!("GPU initialized: {:.0} ms", t_start.elapsed().as_secs_f64() * 1000.0);
        render.upload_scene(&self.scene);
        log::info!("initial scene uploaded: {:.0} ms", t_start.elapsed().as_secs_f64() * 1000.0);

        let (centroid, radius) = scene_bounds(&self.scene);
        let size = window.inner_size();
        let mut camera = Camera::new(
            centroid,
            radius * 2.5,
            Vec2::new(size.width as f32, size.height as f32),
        );
        camera.far  = radius * 20.0;
        camera.near = radius * 0.01;

        // Initialize egui-winit state (must be done before window is moved into AppState)
        self.egui_winit = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,  // theme
            None,  // max_texture_side
        ));

        self.state = AppState::Running { window, render, camera };

        // Execute -c initial commands
        let cmds: Vec<String> = self.initial_commands.drain(..).collect();
        for line in cmds {
            self.run_command_line(&line);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Forward every event to egui first; remember if it was consumed.
        let egui_consumed = if let (Some(egui_state), AppState::Running { window, .. }) =
            (&mut self.egui_winit, &mut self.state)
        {
            egui_state.on_window_event(&**window, &event).consumed
        } else {
            false
        };

        let AppState::Running { window, render, camera } = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                        event_loop.exit();
                    }
                }
            }

            WindowEvent::Resized(size) => {
                render.resize(size.width, size.height);
                camera.resize(size.width as f32, size.height as f32);
                window.request_redraw();
            }

            WindowEvent::MouseInput { state, button, .. } if !egui_consumed => {
                match button {
                    MouseButton::Left => {
                        self.left_pressed = state == ElementState::Pressed;
                        if state == ElementState::Pressed {
                            // Record press position for click-vs-drag detection
                            self.mouse_press_pos = self.last_mouse_pos;
                        } else {
                            // Left button released: check if it's a click (< 5 px movement)
                            let is_click = match (self.mouse_press_pos, self.last_mouse_pos) {
                                (Some(press), Some(release)) => (release - press).length() < 5.0,
                                _ => false,
                            };
                            if is_click {
                                if let AppState::Running { render, window, .. } = &mut self.state {
                                    if let Some(pos) = self.last_mouse_pos {
                                        if Self::handle_pick(render, &self.scene, pos.x as u32, pos.y as u32) {
                                            window.request_redraw();
                                        }
                                    }
                                }
                            }
                            self.mouse_press_pos = None;
                            self.last_mouse_pos = None;
                        }
                    }
                    MouseButton::Right => {
                        self.right_pressed = state == ElementState::Pressed;
                        if state == ElementState::Released {
                            self.last_mouse_pos = None;
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let pos = Vec2::new(position.x as f32, position.y as f32);
                if !egui_consumed {
                    if let Some(last) = self.last_mouse_pos {
                        let delta = pos - last;
                        if self.left_pressed {
                            camera.arcball_rotate(delta);
                            window.request_redraw();
                        } else if self.right_pressed {
                            camera.pan(delta);
                            window.request_redraw();
                        }
                    }
                }
                self.last_mouse_pos = Some(pos);
            }

            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p)   => p.y as f32 * 0.01,
                };
                camera.zoom(scroll);
                window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                render.update_uniforms(camera);

                // ── Build egui frame ─────────────────────────────────────────
                let raw_input = if let Some(state) = &mut self.egui_winit {
                    state.take_egui_input(&**window)
                } else {
                    egui::RawInput::default()
                };

                let mut preset_action: Option<u8> = None;
                let full_output = self.egui_ctx.run(raw_input, |ctx| {
                    egui::TopBottomPanel::bottom("toolbar")
                        .resizable(false)
                        .min_height(40.0)
                        .show(ctx, |ui| {
                            ui.horizontal_centered(|ui| {
                                if ui.button("Default").clicked() {
                                    preset_action = Some(0);
                                }
                                ui.add_space(8.0);
                                if ui.button("Chain Surface").clicked() {
                                    preset_action = Some(1);
                                }
                                ui.add_space(8.0);
                                if ui.button("Binding Site").clicked() {
                                    preset_action = Some(2);
                                }
                                ui.add_space(8.0);
                                if ui.button("Pocket Surface").clicked() {
                                    preset_action = Some(3);
                                }
                                ui.separator();
                                if ui.button("All Reps").clicked() {
                                    preset_action = Some(10);
                                }
                                ui.add_space(4.0);
                                if ui.button("Backbone+Surface").clicked() {
                                    preset_action = Some(11);
                                }
                                ui.add_space(4.0);
                                if ui.button("Lines").clicked() {
                                    preset_action = Some(12);
                                }
                                ui.add_space(4.0);
                                if ui.button("Spectrum").clicked() {
                                    preset_action = Some(14);
                                }
                                ui.add_space(4.0);
                                if ui.button("Neon Glow").clicked() {
                                    preset_action = Some(15);
                                }
                            });
                        });
                });

                if let Some(state) = &mut self.egui_winit {
                    state.handle_platform_output(&**window, full_output.platform_output);
                }

                let pixels_per_point = full_output.pixels_per_point;
                let primitives = self.egui_ctx.tessellate(full_output.shapes, pixels_per_point);
                let screen_desc = ScreenDescriptor {
                    size_in_pixels: [render.config.width, render.config.height],
                    pixels_per_point,
                };

                if let Err(e) = render.render(&primitives, &screen_desc, full_output.textures_delta) {
                    log::error!("Render error: {e}");
                }

                // Apply preset after rendering (scene re-upload on next about_to_wait)
                if let Some(preset) = preset_action {
                    let bg: Option<wgpu::Color> = match preset {
                        10 => Some(wgpu::Color { r: 1.00, g: 1.00, b: 1.00, a: 1.0 }), // white
                        11 => Some(wgpu::Color { r: 0.72, g: 0.72, b: 0.72, a: 1.0 }), // light gray
                        12 => Some(wgpu::Color { r: 0.22, g: 0.22, b: 0.22, a: 1.0 }), // dark gray
                        14 => Some(wgpu::Color { r: 0.05, g: 0.25, b: 0.70, a: 1.0 }), // vivid blue
                        15 => Some(wgpu::Color { r: 0.02, g: 0.02, b: 0.05, a: 1.0 }), // near-black
                        _  => None,
                    };
                    if let Some(c) = bg {
                        render.bg_color = c;
                    }
                    // Surface alpha per preset (Chain Surface intentionally excluded)
                    match preset {
                        3  => render.surface_alpha = 0.75, // Pocket Surface
                        10 => render.surface_alpha = 0.55, // All Reps
                        11 => render.surface_alpha = 0.70, // Backbone+Surface
                        _  => {}
                    }
                    // Bloom & lighting: Neon Glow uses strong bloom; others reset
                    match preset {
                        15 => {
                            render.bloom_threshold = 0.4;
                            render.bloom_intensity = 0.6;
                            render.light_intensity = 1.8;
                            render.ibl_intensity = 1.5;
                            render.edge_strength = 0.0;
                        }
                        _ => {
                            render.bloom_threshold = 1.0;
                            render.bloom_intensity = 0.0;
                            render.light_intensity = 1.0;
                            render.ibl_intensity = 1.0;
                            render.edge_strength = 1.0;
                        }
                    }
                    match preset {
                        0  => apply_default_view(&mut self.scene),
                        1  => apply_chain_surface_view(&mut self.scene),
                        2  => apply_binding_site_view(&mut self.scene),
                        3  => apply_pocket_surface_view(&mut self.scene),
                        10 => apply_all_reps_view(&mut self.scene),
                        11 => apply_backbone_surface_view(&mut self.scene),
                        12 => apply_lines_view(&mut self.scene),
                        14 => apply_spectrum_view(&mut self.scene),
                        15 => apply_neon_glow_view(&mut self.scene),
                        _  => {}
                    }
                    self.scene_dirty = true;
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Process pending commands from prompt thread.
        // Collect quit separately so scene_dirty is flushed before exiting.
        let mut pending_quit = false;
        if let Some(rx) = &self.cmd_rx {
            while let Ok(cmd) = rx.try_recv() {
                if matches!(cmd, Command::Quit) {
                    pending_quit = true;
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(crate::command::CommandResponse::Ok("bye".into()));
                    }
                    continue;
                }

                if let Command::Background(rgb) = cmd {
                    if let AppState::Running { render, window, .. } = &mut self.state {
                        render.bg_color = wgpu::Color {
                            r: rgb[0] as f64,
                            g: rgb[1] as f64,
                            b: rgb[2] as f64,
                            a: 1.0,
                        };
                        window.request_redraw();
                    }
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(crate::command::CommandResponse::Ok(String::new()));
                    }
                    continue;
                }

                if let Command::Set { ref name, value } = cmd {
                    if let AppState::Running { render, window, .. } = &mut self.state {
                        match name.as_str() {
                            "transparency" | "surface_transparency" => {
                                render.surface_alpha = (1.0 - value).clamp(0.0, 1.0);
                            }
                            "edge_strength" => {
                                render.edge_strength = value.max(0.0);
                            }
                            "roughness" => {
                                render.roughness = value.clamp(0.0, 1.0);
                            }
                            "metallic" => {
                                render.metallic = value.clamp(0.0, 1.0);
                            }
                            "ibl_intensity" => {
                                render.ibl_intensity = value.max(0.0);
                            }
                            "shadow_strength" | "shadow" => {
                                render.shadow_strength = value.clamp(0.0, 1.0);
                            }
                            "bloom_threshold" => {
                                render.bloom_threshold = value.max(0.0);
                            }
                            "bloom_intensity" | "bloom" => {
                                render.bloom_intensity = value.max(0.0);
                            }
                            _ => {}
                        }
                        window.request_redraw();
                    }
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(crate::command::CommandResponse::Ok(String::new()));
                    }
                    continue;
                }

                if let Command::SetColor { ref rep, color, ref sel } = cmd {
                    apply_set_color(&mut self.scene, rep, color, sel.as_deref());
                    self.scene_dirty = true;
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(crate::command::CommandResponse::Ok(String::new()));
                    }
                    continue;
                }

                let AppState::Running { camera, .. } = &mut self.state else { break };
                let (response, dirty) = executor::execute(cmd, &mut self.scene, camera);

                if dirty {
                    self.scene_dirty = true;
                }

                if let Some(tx) = &self.resp_tx {
                    let _ = tx.send(response);
                }
            }
        }

        // Re-upload scene if modified (must happen before quit so surface is built)
        if self.scene_dirty {
            if let AppState::Running { render, .. } = &mut self.state {
                render.upload_scene(&self.scene);
            }
            self.scene_dirty = false;
        }

        if pending_quit {
            event_loop.exit();
            return;
        }

        if let AppState::Running { window, .. } = &self.state {
            window.request_redraw();
        }
    }
}

impl App {
    /// Perform a pick at physical pixel (px, py), print info to stdout, and
    /// update the highlight uniform. Returns true if a redraw is needed.
    fn handle_pick(render: &mut RenderState, scene: &Scene, px: u32, py: u32) -> bool {
        match render.pick_at(px, py) {
            Some(PickResult::Atom((obj_name, atom_idx))) => {
                let resid = render.get_residue_id(&obj_name, atom_idx);
                render.set_highlight(resid);
                if let Some(obj) = scene.get(&obj_name) {
                    if let Some(atom) = obj.structure.atoms.get(atom_idx) {
                        println!(
                            "Picked: {} {}:{}{}  {}",
                            atom.residue.name.trim(),
                            atom.residue.chain,
                            atom.residue.seq_num,
                            atom.residue.ins_code.map(|c| c.to_string()).unwrap_or_default(),
                            atom.name.trim(),
                        );
                    }
                }
                true
            }
            Some(PickResult::Residue((obj_name, atom_idx))) => {
                let resid = render.get_residue_id(&obj_name, atom_idx);
                render.set_highlight(resid);
                if let Some(obj) = scene.get(&obj_name) {
                    if let Some(atom) = obj.structure.atoms.get(atom_idx) {
                        println!(
                            "Picked: {} {}:{}{}",
                            atom.residue.name.trim(),
                            atom.residue.chain,
                            atom.residue.seq_num,
                            atom.residue.ins_code.map(|c| c.to_string()).unwrap_or_default(),
                        );
                    }
                }
                true
            }
            None => {
                render.clear_highlight();
                true
            }
        }
    }

    /// Parse and execute a command line synchronously (used for -c and initial commands).
    fn run_command_line(&mut self, line: &str) {
        use crate::command::parser::parse_command;
        match parse_command(line) {
            Ok(cmd) => {
                // Background / Light are handled here rather than in executor (need render access)
                if let Command::Background(rgb) = cmd {
                    if let AppState::Running { render, .. } = &mut self.state {
                        render.bg_color = wgpu::Color {
                            r: rgb[0] as f64,
                            g: rgb[1] as f64,
                            b: rgb[2] as f64,
                            a: 1.0,
                        };
                    }
                    return;
                }
                if let Command::Light { intensity, elevation, azimuth } = cmd {
                    if let AppState::Running { render, window, .. } = &mut self.state {
                        if let Some(v) = intensity { render.light_intensity    = v; }
                        if let Some(v) = elevation { render.light_elevation_deg = v; }
                        if let Some(v) = azimuth   { render.light_azimuth_deg   = v; }
                        window.request_redraw();
                    }
                    return;
                }
                if let Command::Set { ref name, value } = cmd {
                    if let AppState::Running { render, .. } = &mut self.state {
                        match name.as_str() {
                            "transparency" | "surface_transparency" => {
                                render.surface_alpha = (1.0 - value).clamp(0.0, 1.0);
                            }
                            "edge_strength" => {
                                render.edge_strength = value.max(0.0);
                            }
                            "roughness" => {
                                render.roughness = value.clamp(0.0, 1.0);
                            }
                            "metallic" => {
                                render.metallic = value.clamp(0.0, 1.0);
                            }
                            "ibl_intensity" => {
                                render.ibl_intensity = value.max(0.0);
                            }
                            "shadow_strength" | "shadow" => {
                                render.shadow_strength = value.clamp(0.0, 1.0);
                            }
                            "bloom_threshold" => {
                                render.bloom_threshold = value.max(0.0);
                            }
                            "bloom_intensity" | "bloom" => {
                                render.bloom_intensity = value.max(0.0);
                            }
                            _ => {}
                        }
                    }
                    return;
                }
                if let Command::SetColor { ref rep, color, ref sel } = cmd {
                    apply_set_color(&mut self.scene, rep, color, sel.as_deref());
                    self.scene_dirty = true;
                    return;
                }
                let AppState::Running { camera, .. } = &mut self.state else { return };
                let (response, dirty) = executor::execute(cmd, &mut self.scene, camera);
                if dirty { self.scene_dirty = true; }
                match response {
                    CommandResponse::Ok(msg) if !msg.is_empty() => println!("{msg}"),
                    CommandResponse::Error(msg) => eprintln!("Error: {msg}"),
                    _ => {}
                }
            }
            Err(e) => eprintln!("Parse error: {e}"),
        }
    }
}

// ── View presets ─────────────────────────────────────────────────────────────

/// Preset 1: restore the default initial view.
/// - polymer atoms: Ribbon + SS colors
/// - ligands (non-water HETATM): BallAndStick + CPK colors
/// - water: hidden
fn apply_default_view(scene: &mut Scene) {
    use crate::scene::object::{REP_BALL_STICK, REP_RIBBON};
    use crate::structure::atom::SecondaryStructure;
    use crate::util::color::{cpk_color, ss_color};
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];

    for (_, obj) in scene.iter_mut() {
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            let is_water = WATERS.contains(&atom.residue.name.trim());
            obj.atom_rep_show[i] = if is_water {
                0
            } else if obj.structure.is_polymer_atom(atom) {
                REP_RIBBON
            } else {
                REP_BALL_STICK
            };
            obj.atom_colors[i] = if obj.structure.is_polymer_atom(atom) {
                let ss = obj.structure.ss.get(i).copied().unwrap_or(SecondaryStructure::Coil);
                ss_color(ss)
            } else {
                cpk_color(&atom.element)
            };
        }
    }
}

/// Preset 2: Gaussian surface colored by chain + BallAndStick ligands.
/// - polymer atoms: Surface, each chain a distinct color
/// - ligands (non-water HETATM): BallAndStick + CPK colors
/// - water: hidden
fn apply_chain_surface_view(scene: &mut Scene) {
    use crate::scene::object::{REP_BALL_STICK, REP_SURFACE};
    use crate::util::color::{chain_color, cpk_color};
    use std::collections::HashMap;
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];

    // Build a stable global chain → color-index map (alphabetical order).
    let mut chain_index: HashMap<char, usize> = HashMap::new();
    let mut next_idx = 0usize;
    for (_, obj) in scene.iter() {
        for atom in &obj.structure.atoms {
            if obj.structure.is_polymer_atom(atom) {
                let e = chain_index.entry(atom.residue.chain).or_insert_with(|| {
                    let i = next_idx;
                    next_idx += 1;
                    i
                });
                let _ = e;
            }
        }
    }

    for (_, obj) in scene.iter_mut() {
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            let is_water = WATERS.contains(&atom.residue.name.trim());
            obj.atom_rep_show[i] = if is_water {
                0
            } else if obj.structure.is_polymer_atom(atom) {
                REP_SURFACE
            } else {
                REP_BALL_STICK
            };
            obj.atom_colors[i] = if obj.structure.is_polymer_atom(atom) {
                let idx = chain_index.get(&atom.residue.chain).copied().unwrap_or(0);
                chain_color(idx)
            } else {
                cpk_color(&atom.element)
            };
        }
    }
}

/// Collect world-space positions of all ligand (non-water HETATM) atoms across the scene.
fn collect_ligand_positions(scene: &Scene) -> Vec<glam::Vec3> {
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];
    let mut positions = Vec::new();
    for (_, obj) in scene.iter() {
        for atom in &obj.structure.atoms {
            if !atom.is_hetatm { continue; }
            if WATERS.contains(&atom.residue.name.trim()) { continue; }
            positions.push(atom.position);
        }
    }
    positions
}

/// Check if an atom is within `cutoff` Å of any ligand position.
fn is_near_ligand(pos: glam::Vec3, ligand_positions: &[glam::Vec3], cutoff: f32) -> bool {
    let cutoff_sq = cutoff * cutoff;
    ligand_positions.iter().any(|lp| pos.distance_squared(*lp) <= cutoff_sq)
}

/// Build a set of residue keys that have at least one atom near a ligand.
/// Returns a HashSet of (chain, seq_num, ins_code).
fn near_ligand_residues(
    obj: &crate::scene::object::MolecularObject,
    ligand_positions: &[glam::Vec3],
    cutoff: f32,
) -> std::collections::HashSet<(char, i32, Option<char>)> {
    let mut residues = std::collections::HashSet::new();
    for atom in &obj.structure.atoms {
        if !obj.structure.is_polymer_atom(atom) { continue; }
        if is_near_ligand(atom.position, ligand_positions, cutoff) {
            residues.insert((atom.residue.chain, atom.residue.seq_num, atom.residue.ins_code));
        }
    }
    residues
}

/// Preset 3: Binding Site — ligand BallAndStick + nearby protein residues BallAndStick.
/// - Ligand: BallAndStick + CPK colors
/// - Protein within 5 Å of ligand (by residue): BallAndStick + chain color
/// - Remaining protein: Ribbon + dim gray
/// - Water: hidden
fn apply_binding_site_view(scene: &mut Scene) {
    use crate::scene::object::{REP_BALL_STICK, REP_RIBBON};
    use crate::util::color::cpk_color;
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];
    const CUTOFF: f32 = 5.0;
    const DIM_GRAY: [f32; 3] = [0.75, 0.75, 0.75];

    let ligand_positions = collect_ligand_positions(scene);

    // Build chain → color index map (stable across objects)
    let mut chain_index: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    let mut next_idx = 0usize;
    for (_, obj) in scene.iter() {
        for atom in &obj.structure.atoms {
            if obj.structure.is_polymer_atom(atom) {
                chain_index.entry(atom.residue.chain).or_insert_with(|| {
                    let i = next_idx; next_idx += 1; i
                });
            }
        }
    }

    // Collect near-ligand residues per object (need separate pass because borrow rules)
    let near_residues: std::collections::HashMap<String, std::collections::HashSet<(char, i32, Option<char>)>> =
        scene.iter().map(|(name, obj)| {
            (name.clone(), near_ligand_residues(obj, &ligand_positions, CUTOFF))
        }).collect();

    for (name, obj) in scene.iter_mut() {
        let near = near_residues.get(name).cloned().unwrap_or_default();
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            let is_water = WATERS.contains(&atom.residue.name.trim());
            let is_polymer = obj.structure.is_polymer_atom(atom);
            let res_key = (atom.residue.chain, atom.residue.seq_num, atom.residue.ins_code);

            if is_water {
                obj.atom_rep_show[i] = 0;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if !is_polymer {
                // Ligand
                obj.atom_rep_show[i] = REP_BALL_STICK;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if near.contains(&res_key) {
                // Near-ligand protein residue
                obj.atom_rep_show[i] = REP_BALL_STICK;
                let idx = chain_index.get(&atom.residue.chain).copied().unwrap_or(0);
                obj.atom_colors[i] = crate::util::color::chain_color(idx);
            } else {
                // Distant protein
                obj.atom_rep_show[i] = REP_RIBBON;
                obj.atom_colors[i] = DIM_GRAY;
            }
        }
    }
}

/// Preset 4: Pocket Surface — ligand BallAndStick + nearby protein residues as Surface.
/// - Ligand: BallAndStick + CPK colors
/// - Protein within 6 Å of ligand (by residue): Surface + CPK element colors
/// - Remaining protein: Ribbon + dim gray
/// - Water: hidden
fn apply_pocket_surface_view(scene: &mut Scene) {
    use crate::scene::object::{REP_BALL_STICK, REP_RIBBON, REP_SURFACE};
    use crate::util::color::cpk_color;
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];
    const CUTOFF: f32 = 6.0;
    const DIM_GRAY: [f32; 3] = [0.75, 0.75, 0.75];

    let ligand_positions = collect_ligand_positions(scene);

    let near_residues: std::collections::HashMap<String, std::collections::HashSet<(char, i32, Option<char>)>> =
        scene.iter().map(|(name, obj)| {
            (name.clone(), near_ligand_residues(obj, &ligand_positions, CUTOFF))
        }).collect();

    for (name, obj) in scene.iter_mut() {
        let near = near_residues.get(name).cloned().unwrap_or_default();
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            let is_water = WATERS.contains(&atom.residue.name.trim());
            let is_polymer = obj.structure.is_polymer_atom(atom);
            let res_key = (atom.residue.chain, atom.residue.seq_num, atom.residue.ins_code);

            if is_water {
                obj.atom_rep_show[i] = 0;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if !is_polymer {
                obj.atom_rep_show[i] = REP_BALL_STICK;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if near.contains(&res_key) {
                obj.atom_rep_show[i] = REP_SURFACE;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else {
                obj.atom_rep_show[i] = REP_RIBBON;
                obj.atom_colors[i] = DIM_GRAY;
            }
        }
    }
}

/// Preset 5: B-Factor — Ribbon colored by B-factor + ligand BallAndStick.
/// - Protein: Ribbon + B-factor colors (blue=low → white=mid → red=high)
/// - Ligand: BallAndStick + CPK colors
/// - Water: hidden
// ── Dev presets ──────────────────────────────────────────────────────────────

/// Dev preset: All Reps — show Ribbon + Surface + Backbone simultaneously on polymer.
fn apply_all_reps_view(scene: &mut Scene) {
    use crate::scene::object::{REP_BACKBONE, REP_BALL_STICK, REP_RIBBON, REP_SURFACE};
    use crate::structure::atom::SecondaryStructure;
    use crate::util::color::{cpk_color, ss_color};
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];

    for (_, obj) in scene.iter_mut() {
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            let is_water = WATERS.contains(&atom.residue.name.trim());
            if is_water {
                obj.atom_rep_show[i] = 0;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if obj.structure.is_polymer_atom(atom) {
                obj.atom_rep_show[i] = REP_RIBBON | REP_SURFACE | REP_BACKBONE;
                let ss = obj.structure.ss.get(i).copied().unwrap_or(SecondaryStructure::Coil);
                obj.atom_colors[i] = ss_color(ss);
            } else {
                obj.atom_rep_show[i] = REP_BALL_STICK;
                obj.atom_colors[i] = cpk_color(&atom.element);
            }
        }
    }
}

/// Dev preset: Backbone + Surface — Cα trace inside transparent surface.
fn apply_backbone_surface_view(scene: &mut Scene) {
    use crate::scene::object::{REP_BACKBONE, REP_BALL_STICK, REP_SURFACE};
    use crate::util::color::{chain_color, cpk_color};
    use std::collections::HashMap;
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];

    let mut chain_index: HashMap<char, usize> = HashMap::new();
    let mut next_idx = 0usize;
    for (_, obj) in scene.iter() {
        for atom in &obj.structure.atoms {
            if obj.structure.is_polymer_atom(atom) {
                chain_index.entry(atom.residue.chain).or_insert_with(|| {
                    let i = next_idx; next_idx += 1; i
                });
            }
        }
    }

    for (_, obj) in scene.iter_mut() {
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            let is_water = WATERS.contains(&atom.residue.name.trim());
            if is_water {
                obj.atom_rep_show[i] = 0;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if obj.structure.is_polymer_atom(atom) {
                obj.atom_rep_show[i] = REP_BACKBONE | REP_SURFACE;
                let idx = chain_index.get(&atom.residue.chain).copied().unwrap_or(0);
                obj.atom_colors[i] = chain_color(idx);
            } else {
                obj.atom_rep_show[i] = REP_BALL_STICK;
                obj.atom_colors[i] = cpk_color(&atom.element);
            }
        }
    }
}

/// Dev preset: Lines — wireframe bond representation for everything.
fn apply_lines_view(scene: &mut Scene) {
    use crate::scene::object::REP_LINES;
    use crate::util::color::{chain_color, cpk_color};
    use std::collections::HashMap;
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];

    let mut chain_index: HashMap<char, usize> = HashMap::new();
    let mut next_idx = 0usize;
    for (_, obj) in scene.iter() {
        for atom in &obj.structure.atoms {
            if obj.structure.is_polymer_atom(atom) {
                chain_index.entry(atom.residue.chain).or_insert_with(|| {
                    let i = next_idx; next_idx += 1; i
                });
            }
        }
    }

    for (_, obj) in scene.iter_mut() {
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            let is_water = WATERS.contains(&atom.residue.name.trim());
            if is_water {
                obj.atom_rep_show[i] = 0;
            } else {
                obj.atom_rep_show[i] = REP_LINES;
                obj.atom_colors[i] = if obj.structure.is_polymer_atom(atom) {
                    let idx = chain_index.get(&atom.residue.chain).copied().unwrap_or(0);
                    chain_color(idx)
                } else {
                    cpk_color(&atom.element)
                };
            }
        }
    }
}


/// Dev preset: Spectrum — Ribbon with N→C rainbow gradient per chain.
fn apply_spectrum_view(scene: &mut Scene) {
    use crate::scene::object::{REP_BALL_STICK, REP_RIBBON};
    use crate::util::color::cpk_color;
    use std::collections::HashMap;
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];

    // First pass: set representations
    for (_, obj) in scene.iter_mut() {
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            let is_water = WATERS.contains(&atom.residue.name.trim());
            if is_water {
                obj.atom_rep_show[i] = 0;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if obj.structure.is_polymer_atom(atom) {
                obj.atom_rep_show[i] = REP_RIBBON;
                // colors assigned below
            } else {
                obj.atom_rep_show[i] = REP_BALL_STICK;
                obj.atom_colors[i] = cpk_color(&atom.element);
            }
        }
    }

    // Second pass: compute per-chain spectrum gradient for polymer atoms
    for (_, obj) in scene.iter_mut() {
        // Group polymer atoms by chain → sorted by (seq_num, ins_code)
        let mut groups: HashMap<char, Vec<(i32, Option<char>, usize)>> = HashMap::new();
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            if obj.structure.is_polymer_atom(atom) {
                groups.entry(atom.residue.chain).or_default()
                    .push((atom.residue.seq_num, atom.residue.ins_code, i));
            }
        }
        for entries in groups.values_mut() {
            entries.sort_unstable_by_key(|&(seq, ins, _)| (seq, ins));
            let n = entries.len();
            for (rank, &(_, _, atom_idx)) in entries.iter().enumerate() {
                let t = if n > 1 { rank as f32 / (n - 1) as f32 } else { 0.5 };
                obj.atom_colors[atom_idx] = spectrum_color(t);
            }
        }
    }
}

/// Neon Glow preset: bright spectrum colors on dark background, designed for bloom.
fn apply_neon_glow_view(scene: &mut Scene) {
    use crate::scene::object::{REP_BALL_STICK, REP_RIBBON};
    use crate::util::color::cpk_color;
    use std::collections::HashMap;
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];

    // Ribbon for polymer, BallAndStick for ligands, hide water
    for (_, obj) in scene.iter_mut() {
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            let is_water = WATERS.contains(&atom.residue.name.trim());
            if is_water {
                obj.atom_rep_show[i] = 0;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if obj.structure.is_polymer_atom(atom) {
                obj.atom_rep_show[i] = REP_RIBBON;
            } else {
                obj.atom_rep_show[i] = REP_BALL_STICK;
                // Bright neon CPK for ligands
                let c = cpk_color(&atom.element);
                obj.atom_colors[i] = [c[0] * 1.5, c[1] * 1.5, c[2] * 1.5];
            }
        }
    }

    // Bright saturated spectrum colors for polymer (boosted for bloom)
    for (_, obj) in scene.iter_mut() {
        let mut groups: HashMap<char, Vec<(i32, Option<char>, usize)>> = HashMap::new();
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            if obj.structure.is_polymer_atom(atom) {
                groups.entry(atom.residue.chain).or_default()
                    .push((atom.residue.seq_num, atom.residue.ins_code, i));
            }
        }
        for entries in groups.values_mut() {
            entries.sort_unstable_by_key(|&(seq, ins, _)| (seq, ins));
            let n = entries.len();
            for (rank, &(_, _, atom_idx)) in entries.iter().enumerate() {
                let t = if n > 1 { rank as f32 / (n - 1) as f32 } else { 0.5 };
                let c = spectrum_color(t);
                // Boost colors above 1.0 so they trigger bloom
                obj.atom_colors[atom_idx] = [c[0] * 1.6, c[1] * 1.6, c[2] * 1.6];
            }
        }
    }
}

/// HSV rainbow: t=0 → blue (240°), t=1 → red (0°).
fn spectrum_color(t: f32) -> [f32; 3] {
    let h = 240.0 * (1.0 - t.clamp(0.0, 1.0));
    let h6 = h / 60.0;
    let i = h6 as u32;
    let f = h6 - i as f32;
    match i {
        0 => [1.0, f,   0.0],
        1 => [1.0 - f, 1.0, 0.0],
        2 => [0.0, 1.0, f],
        3 => [0.0, 1.0 - f, 1.0],
        4 => [f,   0.0, 1.0],
        _ => [1.0, 0.0, 1.0 - f],
    }
}

/// Apply a per-representation color override to matching objects.
/// `rep`: "surface" or "ribbon".
/// `color`: Some(rgb) to set override, None to reset to per-atom colors.
/// `sel`: optional object name filter; None means all objects.
fn apply_set_color(scene: &mut Scene, rep: &str, color: Option<[f32; 3]>, sel: Option<&str>) {
    for (name, obj) in scene.iter_mut() {
        if let Some(target) = sel {
            if name != target {
                continue;
            }
        }
        match rep {
            "surface" => obj.surface_color_override = color,
            "ribbon"  => obj.ribbon_color_override  = color,
            _ => {}
        }
    }
}

fn scene_bounds(scene: &Scene) -> (glam::Vec3, f32) {
    let mut min = glam::Vec3::splat(f32::MAX);
    let mut max = glam::Vec3::splat(f32::MIN);
    let mut count = 0usize;
    let mut sum = glam::Vec3::ZERO;

    for (_, obj) in scene.iter() {
        for atom in &obj.structure.atoms {
            min = min.min(atom.position);
            max = max.max(atom.position);
            sum += atom.position;
            count += 1;
        }
    }

    if count == 0 {
        return (glam::Vec3::ZERO, 10.0);
    }
    let centroid = sum / count as f32;
    let radius   = ((max - min).length() * 0.5).max(1.0);
    (centroid, radius)
}
