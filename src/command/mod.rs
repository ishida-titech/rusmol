pub mod executor;
pub mod parser;
pub mod prompt;
pub mod selection;

use std::path::PathBuf;

use crate::scene::object::RepresentationType;
pub use selection::SelectionExpr;

#[derive(Debug, Clone)]
pub enum ColorSpec {
    /// Named color (red, blue, …) → RGB
    Rgb([f32; 3]),
    /// Reset to CPK element colors
    Element,
    /// Color by chain index
    Chain,
    /// Color by secondary structure (helix=red, sheet=yellow, coil=white)
    SecondaryStructure,
    /// Rainbow gradient from N-terminus (blue) to C-terminus (red), per chain
    Spectrum,
    /// Color by B-factor (blue=low → white=mid → red=high)
    BFactor,
}

#[derive(Debug, Clone, Copy)]
pub enum TraceAction {
    Next,
    Prev,
    GoTo(usize),
    Quit,
}

#[derive(Debug)]
pub enum Command {
    Load { path: PathBuf, name: Option<String> },
    Select { name: String, expr: SelectionExpr },
    Show { repr: RepresentationType, sel: Option<String> },
    Hide { repr: RepresentationType, sel: Option<String> },
    Color { spec: ColorSpec, sel: Option<String> },
    Enable(String),
    Disable(String),
    Delete(String),
    Zoom { sel: Option<String> },
    Reset,
    Background([f32; 3]),
    /// Adjust light source.  All fields are optional; omitted values are unchanged.
    Light { intensity: Option<f32>, elevation: Option<f32>, azimuth: Option<f32> },
    /// Adjust second light source (PyMOL light2).
    Light2 { intensity: Option<f32>, elevation: Option<f32>, azimuth: Option<f32> },
    /// PyMOL-compatible: `set name, value`
    /// Supported names: transparency, surface_transparency
    Set { name: String, value: f32 },
    /// PyMOL-compatible: `set surface_color|cartoon_color, color[, sel]`
    /// color=None means reset to per-atom colors ("default").
    SetColor { rep: String, color: Option<[f32; 3]>, sel: Option<String> },
    /// `get [name]` — show current parameter value(s)
    Get { name: Option<String> },
    /// Save a screenshot as PNG: `png <filename>`
    Png { path: PathBuf },
    /// Load dock trace: `docktrace <trace_file>, <ligand_pdbqt>`
    DockTrace { trace_path: PathBuf, ligand_path: PathBuf },
    /// Navigate within dock trace mode
    DockTraceNav(TraceAction),
    Help,
    Quit,
}

#[derive(Debug)]
pub enum CommandResponse {
    Ok(String),
    Error(String),
    DockTraceStep { step: usize, total: usize, info: String },
    DockTraceExit,
}
