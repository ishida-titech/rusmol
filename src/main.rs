mod app;
mod cli;
mod command;
mod docktrace;
mod render;
mod scene;
mod structure;
mod util;

use clap::Parser;
use cli::Cli;
use scene::{Scene, object::MolecularObject};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let log_level = if cli.verbose { "info" } else { "warn" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .filter_module("egui_wgpu::renderer", log::LevelFilter::Error)
        .init();

    if cli.files.is_empty() {
        eprintln!(
            "RusMol {}\n\
             Error: no input file specified.\n\n\
             Usage: rusmol <file.pdb|file.pdbqt> [more files ...] [-c \"commands\"]\n\
             Try:   rusmol --help",
            env!("CARGO_PKG_VERSION"),
        );
        std::process::exit(1);
    }

    // ── Startup banner ────────────────────────────────────────────────────
    // The title (name + version, then a one-line description) is boxed between
    // two horizontal rules; each loaded structure follows as a small tree whose
    // rows share a "|" gutter. ASCII only, so it renders on any terminal.
    let rule = "-".repeat(70);
    eprintln!();
    eprintln!("{rule}");
    eprintln!("  RusMol   version {}", env!("CARGO_PKG_VERSION"));
    eprintln!("    a Rust-based lightweight Molecular structure viewer");
    eprintln!("{rule}");
    eprintln!();

    // Load all files into scene
    let mut scene = Scene::new();
    for path in &cli.files {
        let is_pdbqt = path.extension().map_or(false, |e| e.eq_ignore_ascii_case("pdbqt"));
        let parse_result = if is_pdbqt {
            structure::pdb::parse_pdbqt(path)
        } else {
            structure::pdb::parse_pdb(path)
        };
        match parse_result {
            Ok(structure) => {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mol")
                    .to_string();
                let n_atoms = structure.atoms.len();
                let summary = command::executor::structure_summary(&structure);
                scene.add_object(MolecularObject::new(name.clone(), structure));

                eprintln!("  {name}  -  {n_atoms} atoms  ({})", path.display());
                for line in summary.lines() {
                    if line.is_empty() {
                        eprintln!();
                    } else {
                        eprintln!("  {line}");
                    }
                }
                eprintln!();
            }
            Err(e) => {
                eprintln!("  Error loading {}: {e}", path.display());
                std::process::exit(1);
            }
        }
    }

    eprintln!("Type 'help' for the command reference.");

    // Split -c commands into lines
    let initial_commands: Vec<String> = cli
        .command
        .as_deref()
        .unwrap_or("")
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();

    // ── Headless mode: no window, no event loop. Execute -c, export, exit. ──
    if cli.headless {
        return run_headless(scene, initial_commands);
    }

    log::info!("starting event loop");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let proxy = event_loop.create_proxy();

    // Spawn prompt thread (only when stdin is a tty and no -c-only mode)
    let (cmd_rx, resp_tx) = {
        use crossbeam_channel::unbounded;
        let (cmd_tx, cmd_rx)   = unbounded::<command::Command>();
        let (resp_tx, resp_rx) = unbounded::<command::CommandResponse>();

        let prompt_proxy = proxy.clone();
        std::thread::spawn(move || {
            command::prompt::run_prompt(cmd_tx, resp_rx, prompt_proxy);
        });

        (Some(cmd_rx), Some(resp_tx))
    };

    let mut app = app::App::new(scene, cmd_rx, resp_tx, initial_commands);
    event_loop.run_app(&mut app)?;

    Ok(())
}

/// Headless (no-window) runner. Builds an offscreen [`RenderState`], applies the
/// `-c` commands via the shared [`app::apply_command`], writes any `render`
/// outputs to disk, then returns. Never touches winit, so it works with no
/// display server.
fn run_headless(mut scene: Scene, initial_commands: Vec<String>) -> anyhow::Result<()> {
    use app::{apply_command, scene_bounds, CmdOutcome};
    use command::{parser::parse_command, CommandResponse};
    use glam::Vec2;
    use render::{camera::Camera, state::RenderState};
    use scene::SceneDirty;

    if initial_commands.is_empty() {
        eprintln!("Error: --headless requires -c \"...\" with at least one command");
        std::process::exit(1);
    }

    let mut render = pollster::block_on(RenderState::new_headless(1000, 1000))?;
    render.upload_scene(&scene, SceneDirty::ALL);

    // Camera framing matches the windowed `resumed` path (1000×1000 viewport).
    let (centroid, radius) = scene_bounds(&scene);
    let mut camera = Camera::new(centroid, radius * 2.5, Vec2::new(1000.0, 1000.0));
    camera.far = radius * 20.0;
    camera.near = radius * 0.01;

    let mut scene_dirty = SceneDirty::NONE;

    // Flush any pending geometry changes before an export so the
    // capture reflects all preceding commands.
    let flush = |render: &mut RenderState, scene: &Scene, dirty: &mut SceneDirty| {
        if !dirty.is_empty() {
            render.upload_scene(scene, *dirty);
            *dirty = SceneDirty::NONE;
        }
    };

    for line in &initial_commands {
        let cmd = match parse_command(line) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Parse error: {e}");
                continue;
            }
        };
        match apply_command(cmd, &mut scene, &mut render, &mut camera, &mut scene_dirty) {
            CmdOutcome::Handled => {}
            CmdOutcome::Quit => break,
            CmdOutcome::Printed(response) => match response {
                CommandResponse::Ok(msg) if !msg.is_empty() => println!("{msg}"),
                CommandResponse::Error(msg) => eprintln!("Error: {msg}"),
                _ => {}
            },
            CmdOutcome::Render(path, w, h) => {
                flush(&mut render, &scene, &mut scene_dirty);
                // No dims → 1000²; only width → square; both → as given
                // (matches the windowed `render` semantics).
                let out_w = w.unwrap_or(1000);
                let out_h = h.or(w).unwrap_or(1000);
                if let Err(e) = render.export(&mut camera, out_w, out_h, &path) {
                    eprintln!("render: {e}");
                }
            }
        }
    }

    Ok(())
}
