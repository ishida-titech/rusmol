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
    Quit,
}

#[derive(Debug)]
pub enum CommandResponse {
    Ok(String),
    Error(String),
}
