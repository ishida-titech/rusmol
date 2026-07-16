use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "rusmol")]
#[command(version)]
#[command(about = "RusMol — a lightweight molecular structure viewer")]
#[command(long_about = "\
RusMol is a fast, lightweight viewer for protein and nucleic-acid structures,
written in Rust with wgpu. It offers a PyMOL-compatible command set, interactive
3D rendering (ribbon, ball-and-stick, surface, and more), and sub-second startup.

After loading one or more structures, RusMol opens a 3D window and an interactive
`RusMol>` prompt. Type `help` at the prompt for the full command reference.

Examples:
  rusmol protein.pdb
  rusmol receptor.pdbqt ligand.pdbqt
  rusmol complex.pdb -c \"show surface; color chain; png out.png; quit\"")]
pub struct Cli {
    /// Molecular structure file(s) to load (PDB / PDBQT)
    pub files: Vec<PathBuf>,

    /// Run a ';'-separated command string after loading, then keep the prompt open
    #[arg(short = 'c', long, value_name = "COMMAND")]
    pub command: Option<String>,

    /// Print verbose (info-level) diagnostics
    #[arg(short = 'v', long)]
    pub verbose: bool,
}
