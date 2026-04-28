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
                        .with_inner_size(PhysicalSize::new(1200u32, 1600u32)),
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
                                if ui.button("初期表示").clicked() {
                                    preset_action = Some(0);
                                }
                                ui.add_space(8.0);
                                if ui.button("Chain Surface").clicked() {
                                    preset_action = Some(1);
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
                    match preset {
                        0 => apply_default_view(&mut self.scene),
                        1 => apply_chain_surface_view(&mut self.scene),
                        _ => {}
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
                                window.request_redraw();
                            }
                            _ => {}
                        }
                    }
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
                            _ => {}
                        }
                    }
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
