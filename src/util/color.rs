/// CPK color scheme: element symbol → [R, G, B] in 0..=255
pub fn cpk_color(element: &str) -> [f32; 3] {
    match element.trim().to_uppercase().as_str() {
        "H"  => [1.00, 1.00, 1.00],
        "C"  => [0.50, 0.50, 0.50],
        "N"  => [0.13, 0.47, 0.71],
        "O"  => [0.80, 0.16, 0.16],
        "S"  => [1.00, 0.78, 0.20],
        "P"  => [1.00, 0.50, 0.00],
        "F"  => [0.12, 0.94, 0.12],
        "CL" => [0.12, 0.94, 0.12],
        "BR" => [0.65, 0.16, 0.16],
        "I"  => [0.58, 0.00, 0.58],
        "FE" => [0.88, 0.40, 0.20],
        "ZN" => [0.49, 0.50, 0.69],
        "MG" => [0.54, 1.00, 0.00],
        "CA" => [0.24, 1.00, 0.00],
        "MN" => [0.61, 0.48, 0.78],
        "CU" => [0.78, 0.50, 0.20],
        _    => [1.00, 0.08, 0.58], // hot pink for unknown
    }
}

/// Van der Waals radius in Angstroms
pub fn vdw_radius(element: &str) -> f32 {
    match element.trim().to_uppercase().as_str() {
        "H"  => 1.20,
        "C"  => 1.70,
        "N"  => 1.55,
        "O"  => 1.52,
        "S"  => 1.80,
        "P"  => 1.80,
        "F"  => 1.47,
        "CL" => 1.75,
        "BR" => 1.85,
        "I"  => 1.98,
        "FE" => 1.40,
        "ZN" => 1.39,
        "MG" => 1.73,
        "CA" => 2.31,
        _    => 1.70,
    }
}

/// Covalent radius in Angstroms (for bond detection)
pub fn covalent_radius(element: &str) -> f32 {
    match element.trim().to_uppercase().as_str() {
        "H"  => 0.31,
        "C"  => 0.76,
        "N"  => 0.71,
        "O"  => 0.66,
        "S"  => 1.05,
        "P"  => 1.07,
        "F"  => 0.57,
        "CL" => 1.02,
        "BR" => 1.20,
        "I"  => 1.39,
        "FE" => 1.32,
        "ZN" => 1.22,
        "MG" => 1.41,
        "CA" => 1.76,
        "MN" => 1.50,
        "CU" => 1.32,
        _    => 0.77,
    }
}

/// Secondary structure colors (PyMOL-like defaults)
pub fn ss_color(ss: crate::structure::atom::SecondaryStructure) -> [f32; 3] {
    use crate::structure::atom::SecondaryStructure;
    match ss {
        SecondaryStructure::Helix => [0.85, 0.20, 0.20], // red
        SecondaryStructure::Sheet => [0.90, 0.80, 0.10], // yellow
        SecondaryStructure::Coil  => [0.90, 0.90, 0.90], // light gray
    }
}

/// Chain colors: up to 8 chains with distinct colors
pub fn chain_color(chain_idx: usize) -> [f32; 3] {
    const COLORS: [[f32; 3]; 8] = [
        [0.24, 0.71, 0.29], // green
        [0.12, 0.47, 0.71], // blue
        [0.84, 0.15, 0.16], // red
        [1.00, 0.50, 0.00], // orange
        [0.58, 0.40, 0.74], // purple
        [0.55, 0.34, 0.29], // brown
        [0.89, 0.47, 0.76], // pink
        [0.50, 0.50, 0.50], // gray
    ];
    COLORS[chain_idx % COLORS.len()]
}
