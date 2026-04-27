mod app;
mod cli;
mod command;
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
        eprintln!("Error: no input file specified.\nUsage: rusmol <file.pdb> [file2.pdb ...]");
        std::process::exit(1);
    }

    // Load all files into scene
    let mut scene = Scene::new();
    for path in &cli.files {
        match structure::pdb::parse_pdb(path) {
            Ok(structure) => {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mol")
                    .to_string();
                let summary = command::executor::structure_summary(&structure);
                scene.add_object(MolecularObject::new(name.clone(), structure));
                eprintln!("Loaded '{name}' from {}", path.display());
                eprintln!("{summary}");
            }
            Err(e) => {
                eprintln!("Error loading {}: {e}", path.display());
                std::process::exit(1);
            }
        }
    }

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

    // Spawn prompt thread (only when stdin is a tty and no -c-only mode)
    let (cmd_rx, resp_tx) = {
        use crossbeam_channel::unbounded;
        let (cmd_tx, cmd_rx)   = unbounded::<command::Command>();
        let (resp_tx, resp_rx) = unbounded::<command::CommandResponse>();

        std::thread::spawn(move || {
            command::prompt::run_prompt(cmd_tx, resp_rx);
        });

        (Some(cmd_rx), Some(resp_tx))
    };

    log::info!("starting event loop");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = app::App::new(scene, cmd_rx, resp_tx, initial_commands);
    event_loop.run_app(&mut app)?;

    Ok(())
}
