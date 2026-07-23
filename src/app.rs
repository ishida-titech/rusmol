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

use crate::command::{executor, Command, CommandResponse, TraceAction};
use crate::docktrace::DockTrace;
use crate::render::{camera::Camera, state::{PickResult, RenderState}};
use wgpu;
use crate::scene::{Scene, SceneDirty};
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
    middle_pressed:  bool,
    ctrl_pressed:    bool,
    /// Physical pixel position where left button was pressed (for click vs drag)
    mouse_press_pos: Option<Vec2>,

    // Which parts of GPU scene data need re-uploading
    scene_dirty: SceneDirty,

    // egui UI
    egui_ctx:   egui::Context,
    egui_winit: Option<egui_winit::State>,

    /// Deferred quit: wait for pending_screenshot to complete before exiting.
    pending_quit: bool,

    /// Deferred screenshot path: transferred to render.pending_screenshot in
    /// about_to_wait AFTER scene re-upload, so the capture reflects -c changes.
    pending_screenshot_path: Option<std::path::PathBuf>,

    /// Active dock trace session (None when not in trace mode).
    dock_trace: Option<DockTrace>,
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
            middle_pressed: false,
            ctrl_pressed: false,
            mouse_press_pos: None,
            scene_dirty: SceneDirty::NONE,
            egui_ctx:   egui::Context::default(),
            egui_winit: None,
            pending_quit: false,
            pending_screenshot_path: None,
            dock_trace: None,
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
                        .with_title("RusMol")
                        .with_inner_size(PhysicalSize::new(1000u32, 1000u32)),
                )
                .expect("Failed to create window"),
        );
        log::info!("window created: {:.0} ms", t_start.elapsed().as_secs_f64() * 1000.0);

        let mut render = pollster::block_on(RenderState::new(window.clone()))
            .expect("Failed to initialize GPU");
        log::info!("GPU initialized: {:.0} ms", t_start.elapsed().as_secs_f64() * 1000.0);
        render.upload_scene(&self.scene, SceneDirty::ALL);
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

    /// Called by winit when the event loop is shutting down, while the window
    /// and display connection are still valid. We terminate the process here
    /// instead of letting `run_app` return and drop the App: on Linux/Vulkan,
    /// tearing down the wgpu surface/device after winit has closed the X11 or
    /// Wayland connection segfaults inside the driver. Any deferred screenshot
    /// has already been written to disk by the time `exit()` was requested, so
    /// skipping destructors here is safe (the OS reclaims all resources).
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        std::process::exit(0);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Forward every event to egui first.
        if let (Some(egui_state), AppState::Running { window, .. }) =
            (&mut self.egui_winit, &mut self.state)
        {
            let _ = egui_state.on_window_event(&**window, &event);
        }
        // Determine if the pointer is over the egui toolbar by checking the
        // physical Y coordinate against the window height.  Both
        // on_window_event().consumed and is_pointer_over_area() are unreliable
        // in release builds (egui consumes all events regardless of pointer
        // position), so we use a direct geometric check.
        let egui_consumed = if let AppState::Running { window, .. } = &self.state {
            const TOOLBAR_LOGICAL_HEIGHT: f32 = 48.0;
            let scale = window.scale_factor() as f32;
            let win_h = window.inner_size().height as f32;
            let toolbar_px = TOOLBAR_LOGICAL_HEIGHT * scale;
            self.last_mouse_pos
                .map(|p| p.y > win_h - toolbar_px)
                .unwrap_or(false)
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

            WindowEvent::ModifiersChanged(mods) => {
                self.ctrl_pressed = mods.state().control_key();
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
                    MouseButton::Middle => {
                        self.middle_pressed = state == ElementState::Pressed;
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
                        let light_drag = self.middle_pressed
                            || (self.left_pressed && self.ctrl_pressed);
                        if light_drag {
                            // Ctrl+left-drag or middle-drag: move light direction
                            let sensitivity = 0.5;
                            render.light_azimuth_deg   += delta.x * sensitivity;
                            render.light_elevation_deg =
                                (render.light_elevation_deg - delta.y * sensitivity)
                                    .clamp(-90.0, 90.0);
                            window.request_redraw();
                        } else if self.left_pressed {
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
                    // Pocket Surface uses a semi-transparent surface and keeps
                    // only the ligand-facing side; other presets leave the
                    // current transparency untouched and show the full surface.
                    render.surface_clip_to_ligand = preset == 3;
                    if preset == 3 {
                        render.surface_alpha = 0.75;
                    }

                    // Reset post-processing and lighting to their defaults.
                    render.bloom_threshold = 1.0;
                    render.bloom_intensity = 0.0;
                    render.light_intensity = 1.0;
                    render.ibl_intensity = 1.0;
                    render.edge_strength = 1.0;
                    render.light2_intensity = 0.0;
                    render.light_elevation_deg = 30.0;
                    render.light_azimuth_deg = 30.0;

                    match preset {
                        0 => apply_default_view(&mut self.scene),
                        1 => apply_chain_surface_view(&mut self.scene),
                        2 => apply_binding_site_view(&mut self.scene),
                        3 => apply_pocket_surface_view(&mut self.scene),
                        _ => {}
                    }
                    // Hydrogen-bond dashes are shown only in the Binding Site view.
                    render.hbond_segments = if preset == 2 {
                        detect_ligand_hbonds(&self.scene)
                    } else {
                        Vec::new()
                    };
                    self.scene_dirty = SceneDirty::ALL;
                }
            }

            _ => {
                // When pointer events land on the egui toolbar, we still need a
                // redraw so that egui_ctx.run() can process the interaction.
                if egui_consumed {
                    if let AppState::Running { window, .. } = &self.state {
                        window.request_redraw();
                    }
                }
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Process pending commands from prompt thread.
        // Collect quit separately so scene_dirty is flushed before exiting.
        let mut pending_quit = false;
        let commands: Vec<Command> = self.cmd_rx.as_ref()
            .map(|rx| rx.try_iter().collect())
            .unwrap_or_default();
        for cmd in commands {
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
                    let mut need_rebuild = false;
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
                            "surface_type" => {
                                use crate::render::surface::SurfaceType;
                                let new_type = if value < 0.5 { SurfaceType::Gaussian } else { SurfaceType::Ses };
                                if render.surface_type != new_type {
                                    render.surface_type = new_type;
                                    need_rebuild = true;
                                }
                            }
                            "surface_quality" => {
                                let new_q = value.clamp(0.2, 2.0);
                                if (render.surface_quality - new_q).abs() > 0.01 {
                                    render.surface_quality = new_q;
                                    need_rebuild = true;
                                }
                            }
                            "surface_smooth" => {
                                let new_s = value.round().clamp(0.0, 100.0) as u32;
                                if render.surface_smooth != new_s {
                                    render.surface_smooth = new_s;
                                    need_rebuild = true;
                                }
                            }
                            "light_intensity"  => render.light_intensity     = value.max(0.0),
                            "light_elevation"  => render.light_elevation_deg = value.clamp(-90.0, 90.0),
                            "light_azimuth"    => render.light_azimuth_deg   = value,
                            "light2_intensity" => render.light2_intensity     = value.max(0.0),
                            "light2_elevation" => render.light2_elevation_deg = value.clamp(-90.0, 90.0),
                            "light2_azimuth"   => render.light2_azimuth_deg   = value,
                            _ => {}
                        }
                        window.request_redraw();
                    }
                    if need_rebuild {
                        self.scene_dirty |= SceneDirty::SURFACE;
                    }
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(crate::command::CommandResponse::Ok(String::new()));
                    }
                    continue;
                }

                if let Command::Get { ref name } = cmd {
                    let msg = if let AppState::Running { render, .. } = &self.state {
                        format_get_params(render, name.as_deref())
                    } else {
                        "Not initialized".to_string()
                    };
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(crate::command::CommandResponse::Ok(msg));
                    }
                    continue;
                }

                if let Command::Light { intensity, elevation, azimuth } = cmd {
                    if let AppState::Running { render, window, .. } = &mut self.state {
                        if let Some(v) = intensity { render.light_intensity     = v; }
                        if let Some(v) = elevation { render.light_elevation_deg = v; }
                        if let Some(v) = azimuth   { render.light_azimuth_deg   = v; }
                        window.request_redraw();
                    }
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(crate::command::CommandResponse::Ok(String::new()));
                    }
                    continue;
                }

                if let Command::Light2 { intensity, elevation, azimuth } = cmd {
                    if let AppState::Running { render, window, .. } = &mut self.state {
                        if let Some(v) = intensity { render.light2_intensity     = v; }
                        if let Some(v) = elevation { render.light2_elevation_deg = v; }
                        if let Some(v) = azimuth   { render.light2_azimuth_deg   = v; }
                        window.request_redraw();
                    }
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(crate::command::CommandResponse::Ok(String::new()));
                    }
                    continue;
                }

                if let Command::SetColor { ref rep, color, ref sel } = cmd {
                    apply_set_color(&mut self.scene, rep, color, sel.as_deref());
                    self.scene_dirty |= dirty_for_set_color(rep);
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(crate::command::CommandResponse::Ok(String::new()));
                    }
                    continue;
                }

                if let Command::Png { path } = cmd {
                    self.pending_screenshot_path = Some(path);
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(crate::command::CommandResponse::Ok(String::new()));
                    }
                    continue;
                }

                if let Command::DockTrace { trace_path, ligand_path } = cmd {
                    let resp = self.handle_docktrace_load(&trace_path, &ligand_path);
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(resp);
                    }
                    continue;
                }

                if let Command::DockTraceNav(action) = cmd {
                    let resp = self.handle_docktrace_nav(action);
                    if let Some(tx) = &self.resp_tx {
                        let _ = tx.send(resp);
                    }
                    continue;
                }

                let AppState::Running { camera, .. } = &mut self.state else { break };
                let (response, dirty) = executor::execute(cmd, &mut self.scene, camera);

                self.scene_dirty |= dirty;

                if let Some(tx) = &self.resp_tx {
                    let _ = tx.send(response);
                }
        }

        // Re-upload scene if modified (must happen before quit so surface is built)
        if !self.scene_dirty.is_empty() {
            if let AppState::Running { render, window, .. } = &mut self.state {
                render.upload_scene(&self.scene, self.scene_dirty);
                window.request_redraw();
            }
            self.scene_dirty = SceneDirty::NONE;
        }

        // Transfer deferred screenshot to render AFTER scene re-upload
        if let Some(path) = self.pending_screenshot_path.take() {
            if let AppState::Running { render, window, .. } = &mut self.state {
                render.pending_screenshot = Some(path);
                window.request_redraw();
            }
        }

        if pending_quit {
            self.pending_quit = true;
        }

        // Deferred quit: wait until pending_screenshot is consumed
        if self.pending_quit {
            let screenshot_done = match &self.state {
                AppState::Running { render, .. } => render.pending_screenshot.is_none(),
                _ => true,
            };
            if screenshot_done {
                event_loop.exit();
                return;
            } else if let AppState::Running { window, .. } = &self.state {
                window.request_redraw();
            }
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
                // Quit via -c: defer to about_to_wait so screenshots can complete
                if matches!(cmd, Command::Quit) {
                    self.pending_quit = true;
                    return;
                }
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
                if let Command::Light2 { intensity, elevation, azimuth } = cmd {
                    if let AppState::Running { render, window, .. } = &mut self.state {
                        if let Some(v) = intensity { render.light2_intensity    = v; }
                        if let Some(v) = elevation { render.light2_elevation_deg = v; }
                        if let Some(v) = azimuth   { render.light2_azimuth_deg   = v; }
                        window.request_redraw();
                    }
                    return;
                }
                if let Command::Set { ref name, value } = cmd {
                    let mut need_rebuild = false;
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
                            "surface_type" => {
                                use crate::render::surface::SurfaceType;
                                let new_type = if value < 0.5 { SurfaceType::Gaussian } else { SurfaceType::Ses };
                                if render.surface_type != new_type {
                                    render.surface_type = new_type;
                                    need_rebuild = true;
                                }
                            }
                            "surface_quality" => {
                                let new_q = value.clamp(0.2, 2.0);
                                if (render.surface_quality - new_q).abs() > 0.01 {
                                    render.surface_quality = new_q;
                                    need_rebuild = true;
                                }
                            }
                            "surface_smooth" => {
                                let new_s = value.round().clamp(0.0, 100.0) as u32;
                                if render.surface_smooth != new_s {
                                    render.surface_smooth = new_s;
                                    need_rebuild = true;
                                }
                            }
                            "light_intensity"  => render.light_intensity     = value.max(0.0),
                            "light_elevation"  => render.light_elevation_deg = value.clamp(-90.0, 90.0),
                            "light_azimuth"    => render.light_azimuth_deg   = value,
                            "light2_intensity" => render.light2_intensity     = value.max(0.0),
                            "light2_elevation" => render.light2_elevation_deg = value.clamp(-90.0, 90.0),
                            "light2_azimuth"   => render.light2_azimuth_deg   = value,
                            _ => {}
                        }
                    }
                    if need_rebuild {
                        self.scene_dirty |= SceneDirty::SURFACE;
                    }
                    return;
                }
                if let Command::SetColor { ref rep, color, ref sel } = cmd {
                    apply_set_color(&mut self.scene, rep, color, sel.as_deref());
                    self.scene_dirty |= dirty_for_set_color(rep);
                    return;
                }
                if let Command::Png { path } = cmd {
                    self.pending_screenshot_path = Some(path);
                    return;
                }
                let AppState::Running { camera, .. } = &mut self.state else { return };
                let (response, dirty) = executor::execute(cmd, &mut self.scene, camera);
                self.scene_dirty |= dirty;
                match response {
                    CommandResponse::Ok(msg) if !msg.is_empty() => println!("{msg}"),
                    CommandResponse::Error(msg) => eprintln!("Error: {msg}"),
                    _ => {}
                }
            }
            Err(e) => eprintln!("Parse error: {e}"),
        }
    }

    fn handle_docktrace_load(
        &mut self,
        trace_path: &std::path::Path,
        ligand_path: &std::path::Path,
    ) -> CommandResponse {
        let dt = match DockTrace::load(trace_path, ligand_path) {
            Ok(dt) => dt,
            Err(e) => return CommandResponse::Error(format!("docktrace: {e}")),
        };

        let ligand_structure = match crate::structure::pdb::parse_pdbqt(ligand_path) {
            Ok(s) => s,
            Err(e) => return CommandResponse::Error(format!("docktrace ligand: {e}")),
        };

        let positions = dt.reconstruct_positions();
        let info = dt.step_info();
        let step = dt.current_step;
        let total = dt.total_steps();

        let mut structure = ligand_structure;
        for (i, pos) in positions.iter().enumerate() {
            if i < structure.atoms.len() {
                structure.atoms[i].position = *pos;
            }
        }

        let obj_name = "_docktrace_ligand".to_string();
        self.scene.remove(&obj_name);

        use crate::scene::object::{MolecularObject, REP_BALL_STICK};
        use crate::util::color::cpk_color;
        let mut obj = MolecularObject::new(obj_name.clone(), structure);
        for (i, atom) in obj.structure.atoms.iter().enumerate() {
            obj.atom_rep_show[i] = REP_BALL_STICK;
            obj.atom_colors[i] = cpk_color(&atom.element);
        }
        self.scene.add_object(obj);

        // Add box wireframe
        self.add_docktrace_box(&dt.header);

        // Zoom camera to ligand + box
        if let AppState::Running { camera, window, .. } = &mut self.state {
            let center = dt.header.box_center;
            let half = dt.header.box_size * 0.5;
            let radius = half.length().max(5.0);
            camera.center = center;
            camera.distance = radius * 3.0;
            window.request_redraw();
        }

        self.scene_dirty = SceneDirty::ALL;
        self.dock_trace = Some(dt);

        CommandResponse::DockTraceStep { step, total, info }
    }

    fn handle_docktrace_nav(&mut self, action: TraceAction) -> CommandResponse {
        let dt = match &mut self.dock_trace {
            Some(dt) => dt,
            None => return CommandResponse::Error("not in dock trace mode".into()),
        };

        match action {
            TraceAction::Next => {
                if !dt.next() {
                    return CommandResponse::Error("already at last step".into());
                }
            }
            TraceAction::Prev => {
                if !dt.prev() {
                    return CommandResponse::Error("already at first step".into());
                }
            }
            TraceAction::GoTo(row) => {
                if row == 0 || row > dt.total_steps() {
                    return CommandResponse::Error(format!("row out of range (1-{})", dt.total_steps()));
                }
                dt.current_step = row - 1;
            }
            TraceAction::Quit => {
                self.scene.remove("_docktrace_ligand");
                self.scene.remove("_docktrace_box");
                self.dock_trace = None;
                self.scene_dirty = SceneDirty::ALL;
                if let AppState::Running { window, .. } = &self.state {
                    window.request_redraw();
                }
                return CommandResponse::DockTraceExit;
            }
        }

        let positions = dt.reconstruct_positions();
        let info = dt.step_info();
        let step = dt.current_step;
        let total = dt.total_steps();

        if let Some(obj) = self.scene.get_mut("_docktrace_ligand") {
            for (i, pos) in positions.iter().enumerate() {
                if i < obj.structure.atoms.len() {
                    obj.structure.atoms[i].position = *pos;
                }
            }
        }

        self.scene_dirty |= SceneDirty::ATOMS;
        if let AppState::Running { window, .. } = &self.state {
            window.request_redraw();
        }

        CommandResponse::DockTraceStep { step, total, info }
    }

    fn add_docktrace_box(&mut self, header: &crate::docktrace::TraceHeader) {
        use crate::structure::atom::{Atom, Bond, ResidueId, Structure};
        use crate::scene::object::{MolecularObject, REP_LINES};

        let center = header.box_center;
        let half = header.box_size * 0.5;

        let corners = [
            center + glam::Vec3::new(-half.x, -half.y, -half.z),
            center + glam::Vec3::new( half.x, -half.y, -half.z),
            center + glam::Vec3::new( half.x,  half.y, -half.z),
            center + glam::Vec3::new(-half.x,  half.y, -half.z),
            center + glam::Vec3::new(-half.x, -half.y,  half.z),
            center + glam::Vec3::new( half.x, -half.y,  half.z),
            center + glam::Vec3::new( half.x,  half.y,  half.z),
            center + glam::Vec3::new(-half.x,  half.y,  half.z),
        ];

        let edges: [(usize, usize); 12] = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ];

        let res_id = ResidueId {
            chain: 'Z',
            seq_num: 999,
            ins_code: None,
            name: "BOX".to_string(),
        };

        let atoms: Vec<Atom> = corners.iter().enumerate().map(|(i, &pos)| Atom {
            serial: (i + 1) as u32,
            name: format!(" X{:<2}", i + 1),
            alt_loc: None,
            residue: res_id.clone(),
            position: pos,
            temp_factor: 0.0,
            element: "X".to_string(),
            is_hetatm: true,
        }).collect();

        let bonds: Vec<Bond> = edges.iter().map(|&(a, b)| Bond { atom1: a, atom2: b }).collect();

        let structure = Structure {
            atoms,
            bonds,
            ss: vec![],
            ..Default::default()
        };

        let obj_name = "_docktrace_box".to_string();
        self.scene.remove(&obj_name);

        let mut obj = MolecularObject::new(obj_name, structure);
        for i in 0..obj.atom_rep_show.len() {
            obj.atom_rep_show[i] = REP_LINES;
            obj.atom_colors[i] = [0.5, 0.5, 0.5];
        }
        self.scene.add_object(obj);
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
/// Return the residue name of the largest ligand (most atoms), or None if no ligands.
fn largest_ligand_resn(scene: &Scene) -> Option<String> {
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];
    let mut count_by_resn: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (_, obj) in scene.iter() {
        for atom in &obj.structure.atoms {
            if !atom.is_hetatm { continue; }
            let resn = atom.residue.name.trim().to_string();
            if WATERS.contains(&resn.as_str()) { continue; }
            *count_by_resn.entry(resn).or_insert(0) += 1;
        }
    }
    count_by_resn.into_iter().max_by_key(|(_, c)| *c).map(|(name, _)| name)
}

/// Collect positions of the largest ligand(s) in the scene.
/// "Largest" = the residue name with the most atoms across all HETATM non-water
/// residues. All instances of that residue name are included.
fn collect_ligand_positions(scene: &Scene) -> Vec<glam::Vec3> {
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];
    // Count atoms per residue name
    let mut count_by_resn: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (_, obj) in scene.iter() {
        for atom in &obj.structure.atoms {
            if !atom.is_hetatm { continue; }
            let resn = atom.residue.name.trim().to_string();
            if WATERS.contains(&resn.as_str()) { continue; }
            *count_by_resn.entry(resn).or_insert(0) += 1;
        }
    }
    let largest_resn = match count_by_resn.iter().max_by_key(|(_, &c)| c) {
        Some((name, _)) => name.clone(),
        None => return Vec::new(),
    };
    let mut positions = Vec::new();
    for (_, obj) in scene.iter() {
        for atom in &obj.structure.atoms {
            if !atom.is_hetatm { continue; }
            if atom.residue.name.trim() != largest_resn { continue; }
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

/// A polar atom that can take part in a hydrogen bond, with the positions of any
/// covalently bonded hydrogens (empty when the structure has no explicit H).
struct PolarAtom {
    pos: glam::Vec3,
    h_pos: Vec<glam::Vec3>,
}

/// True if `donor` (bearing a hydrogen) forms a hydrogen bond to `acceptor`:
/// some H sits within `max_ha` of the acceptor with a D–H···A angle ≥ 120°.
fn is_hbond_geometry(donor: &PolarAtom, acceptor: &PolarAtom, max_ha: f32) -> bool {
    const MAX_COS: f32 = -0.5; // cos(120°): D–H···A must be at least 120°
    for &h in &donor.h_pos {
        if h.distance(acceptor.pos) > max_ha { continue; }
        let to_d = (donor.pos - h).normalize_or_zero();
        let to_a = (acceptor.pos - h).normalize_or_zero();
        if to_d.dot(to_a) <= MAX_COS { return true; }
    }
    false
}

/// Detect hydrogen bonds between the target ligand and the protein for the
/// Binding Site view. Candidate atoms are electronegative (N/O/S/F). When
/// hydrogens are present the D–H···A geometry is checked; otherwise a plain
/// donor···acceptor heavy-atom distance is used. Returns heavy-atom endpoint
/// pairs to be drawn as dashed lines.
fn detect_ligand_hbonds(scene: &Scene) -> Vec<(glam::Vec3, glam::Vec3)> {
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];
    const MAX_DA: f32 = 3.5; // donor···acceptor heavy-atom distance
    const MIN_DA: f32 = 2.4;
    const MAX_HA: f32 = 2.7; // H···acceptor distance

    let Some(target) = largest_ligand_resn(scene) else { return Vec::new() };

    let mut lig: Vec<PolarAtom> = Vec::new();
    let mut pro: Vec<PolarAtom> = Vec::new();

    for (_, obj) in scene.iter() {
        let atoms = &obj.structure.atoms;
        // Map heavy-atom index → positions of its bonded hydrogens.
        let mut h_of: std::collections::HashMap<usize, Vec<glam::Vec3>> = std::collections::HashMap::new();
        for b in &obj.structure.bonds {
            let (i, j) = (b.atom1, b.atom2);
            if i >= atoms.len() || j >= atoms.len() { continue; }
            if atoms[i].element == "H" { h_of.entry(j).or_default().push(atoms[i].position); }
            if atoms[j].element == "H" { h_of.entry(i).or_default().push(atoms[j].position); }
        }
        for (i, atom) in atoms.iter().enumerate() {
            if !matches!(atom.element.as_str(), "N" | "O" | "S" | "F") { continue; }
            let is_water = WATERS.contains(&atom.residue.name.trim());
            let is_ligand = atom.is_hetatm && !is_water && atom.residue.name.trim() == target;
            let is_protein = obj.structure.is_polymer_atom(atom);
            if !is_ligand && !is_protein { continue; }
            let p = PolarAtom {
                pos: atom.position,
                h_pos: h_of.get(&i).cloned().unwrap_or_default(),
            };
            if is_ligand { lig.push(p); } else { pro.push(p); }
        }
    }

    // If no hydrogens are present anywhere, fall back to distance-only pairing.
    let any_h = lig.iter().chain(pro.iter()).any(|p| !p.h_pos.is_empty());

    let mut out = Vec::new();
    for l in &lig {
        for p in &pro {
            let d = l.pos.distance(p.pos);
            if !(MIN_DA..=MAX_DA).contains(&d) { continue; }
            let hbond = if any_h {
                is_hbond_geometry(l, p, MAX_HA) || is_hbond_geometry(p, l, MAX_HA)
            } else {
                true // both are electronegative and within donor···acceptor range
            };
            if hbond {
                out.push((l.pos, p.pos));
            }
        }
    }
    out
}

/// Preset 3: Binding Site — ligand BallAndStick + nearby protein residues BallAndStick.
/// - Ligand: BallAndStick + CPK colors
/// - Protein within 4 Å of ligand (by residue): BallAndStick + chain color
/// - Remaining protein: Ribbon + dim gray
/// - Water: hidden
fn apply_binding_site_view(scene: &mut Scene) {
    use crate::scene::object::{REP_BALL_STICK, REP_RIBBON, REP_STICK};
    use crate::util::color::cpk_color;
    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];
    const CUTOFF: f32 = 4.0;
    // Match the darkened-CPK carbon of the binding-site sticks so the distant
    // ribbon reads as almost the same tone (cpk carbon 0.5 × DARKEN 0.25 = 0.125).
    const DIM_GRAY: [f32; 3] = [0.125, 0.125, 0.125];

    let ligand_positions = collect_ligand_positions(scene);
    let target_resn = largest_ligand_resn(scene);

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

            if atom.element == "H" {
                // Hide hydrogens — heavy atoms only for a clean binding-site view.
                obj.atom_rep_show[i] = 0;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if is_water {
                obj.atom_rep_show[i] = 0;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if !is_polymer && target_resn.as_deref() == Some(atom.residue.name.trim()) {
                // Target ligand
                obj.atom_rep_show[i] = REP_BALL_STICK;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if !is_polymer {
                // Other ligands / ions — hide
                obj.atom_rep_show[i] = 0;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if near.contains(&res_key) {
                // Near-ligand protein residue — plain sticks, darkened CPK so the
                // protein reads as distinct from the full-brightness ligand.
                obj.atom_rep_show[i] = REP_STICK;
                obj.atom_colors[i] = crate::util::color::cpk_dark(&atom.element);
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
    let target_resn = largest_ligand_resn(scene);

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
            } else if !is_polymer && target_resn.as_deref() == Some(atom.residue.name.trim()) {
                obj.atom_rep_show[i] = REP_BALL_STICK;
                obj.atom_colors[i] = cpk_color(&atom.element);
            } else if !is_polymer {
                obj.atom_rep_show[i] = 0;
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

/// Apply a per-representation color override to matching objects.
/// `rep`: "surface" or "ribbon".
/// `color`: Some(rgb) to set override, None to reset to per-atom colors.
/// `sel`: optional object name filter; None means all objects.
fn dirty_for_set_color(rep: &str) -> SceneDirty {
    match rep {
        "surface" => SceneDirty::SURFACE,
        "ribbon"  => SceneDirty::ATOMS | SceneDirty::RIBBON,
        _         => SceneDirty::ALL,
    }
}

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

/// Format parameter values for the `get` command.
fn format_get_params(render: &RenderState, name: Option<&str>) -> String {
    use crate::render::surface::SurfaceType;

    let surface_type_str = match render.surface_type {
        SurfaceType::Gaussian => "gaussian",
        SurfaceType::Ses => "ses",
    };

    let params: &[(&str, String)] = &[
        ("transparency",    format!("{:.2}", 1.0 - render.surface_alpha)),
        ("surface_type",    surface_type_str.to_string()),
        ("surface_quality", format!("{:.2}", render.surface_quality)),
        ("edge_strength",   format!("{:.2}", render.edge_strength)),
        ("roughness",       format!("{:.2}", render.roughness)),
        ("metallic",        format!("{:.2}", render.metallic)),
        ("ibl_intensity",   format!("{:.2}", render.ibl_intensity)),
        ("shadow_strength", format!("{:.2}", render.shadow_strength)),
        ("bloom_threshold", format!("{:.2}", render.bloom_threshold)),
        ("bloom_intensity", format!("{:.2}", render.bloom_intensity)),
        ("light_intensity",  format!("{:.2}", render.light_intensity)),
        ("light_elevation",  format!("{:.1}", render.light_elevation_deg)),
        ("light_azimuth",    format!("{:.1}", render.light_azimuth_deg)),
        ("light2_intensity", format!("{:.2}", render.light2_intensity)),
        ("light2_elevation", format!("{:.1}", render.light2_elevation_deg)),
        ("light2_azimuth",   format!("{:.1}", render.light2_azimuth_deg)),
    ];

    match name {
        None => {
            // Show all
            let max_name_len = params.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
            params
                .iter()
                .map(|(n, v)| format!("  {:width$} = {}", n, v, width = max_name_len))
                .collect::<Vec<_>>()
                .join("\n")
        }
        Some(query) => {
            if let Some((_, v)) = params.iter().find(|(n, _)| *n == query) {
                format!("{query} = {v}")
            } else {
                format!("unknown parameter: '{query}'")
            }
        }
    }
}
