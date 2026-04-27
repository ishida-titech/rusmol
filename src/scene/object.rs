use crate::structure::atom::{SecondaryStructure, Structure};
use crate::util::color::{cpk_color, ss_color};

// ── Per-atom representation bit flags ────────────────────────────────────────
pub const REP_BALL_STICK: u8 = 0b00001;
pub const REP_BACKBONE:   u8 = 0b00010;
pub const REP_RIBBON:     u8 = 0b00100;
pub const REP_SURFACE:    u8 = 0b01000;
pub const REP_LINES:      u8 = 0b10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentationType {
    BallAndStick,
    Backbone,
    Ribbon,
    Surface,
    Lines,
}

impl RepresentationType {
    pub fn to_bit(self) -> u8 {
        match self {
            RepresentationType::BallAndStick => REP_BALL_STICK,
            RepresentationType::Backbone     => REP_BACKBONE,
            RepresentationType::Ribbon       => REP_RIBBON,
            RepresentationType::Surface      => REP_SURFACE,
            RepresentationType::Lines        => REP_LINES,
        }
    }
}

const WATERS: &[&str] = &["HOH", "WAT", "DOD"];

#[derive(Debug)]
pub struct MolecularObject {
    pub name: String,
    pub structure: Structure,
    pub visible: bool,
    /// Per-atom representation bitmask (same length as `structure.atoms`).
    /// Each bit corresponds to a REP_* constant.
    pub atom_rep_show: Vec<u8>,
    /// Per-atom display colors (RGB).
    pub atom_colors: Vec<[f32; 3]>,
}

impl MolecularObject {
    pub fn new(name: String, structure: Structure) -> Self {
        let atom_rep_show = structure
            .atoms
            .iter()
            .map(|a| {
                if WATERS.contains(&a.residue.name.trim()) {
                    0u8
                } else if structure.is_polymer_atom(a) {
                    REP_RIBBON
                } else {
                    REP_BALL_STICK
                }
            })
            .collect();

        let atom_colors = structure
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if structure.is_polymer_atom(a) {
                    let ss = structure.ss.get(i).copied().unwrap_or(SecondaryStructure::Coil);
                    ss_color(ss)
                } else {
                    cpk_color(&a.element)
                }
            })
            .collect();

        Self {
            name,
            structure,
            visible: true,
            atom_rep_show,
            atom_colors,
        }
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Returns true if any atom has the given representation active.
    pub fn has_representation(&self, rep: RepresentationType) -> bool {
        let bit = rep.to_bit();
        self.atom_rep_show.iter().any(|&f| f & bit != 0)
    }
}
