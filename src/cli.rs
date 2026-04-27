use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "rusmol")]
#[command(version)]
#[command(about = "Lightweight molecular structure viewer")]
pub struct Cli {
    /// Molecular structure file(s) to load (PDB format)
    pub files: Vec<PathBuf>,

    /// Execute command string after loading
    #[arg(short = 'c', long, value_name = "COMMAND")]
    pub command: Option<String>,

    /// Enable verbose output
    #[arg(short = 'v', long)]
    pub verbose: bool,
}
