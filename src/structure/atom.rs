use glam::Vec3;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SecondaryStructure {
    #[default]
    Coil,
    Helix,
    Sheet,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResidueId {
    pub chain: char,
    pub seq_num: i32,
    pub ins_code: Option<char>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub serial: u32,
    /// Atom name as in PDB (e.g. " CA ", "N   ")
    pub name: String,
    pub alt_loc: Option<char>,
    pub residue: ResidueId,
    pub position: Vec3,
    pub occupancy: f32,
    pub temp_factor: f32,
    /// Element symbol, uppercase (e.g. "C", "N", "CA" for calcium)
    pub element: String,
    pub is_hetatm: bool,
}

impl Atom {
    /// Returns the trimmed atom name (e.g. "CA", "N", "OG1")
    pub fn name_trimmed(&self) -> &str {
        self.name.trim()
    }
}

#[derive(Debug, Clone)]
pub struct Bond {
    pub atom1: usize,
    pub atom2: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Structure {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    /// Indices into atoms grouped by chain (chain_char → start..end)
    pub chain_ranges: HashMap<char, std::ops::Range<usize>>,
    /// Per-atom secondary structure assignment (same length as atoms).
    pub ss: Vec<SecondaryStructure>,
    /// COMPND: chain letter → molecule name
    pub compnd: HashMap<char, String>,
    /// HETNAM: het_id → chemical name
    pub hetnam: HashMap<String, String>,
    /// HETSYN: het_id → first synonym
    pub hetsyn: HashMap<String, String>,
    /// SEQRES: chain → set of residue names that are part of the polymer chain.
    /// Populated from SEQRES records; empty when the PDB file omits them.
    pub seqres: HashMap<char, HashSet<String>>,

    /// Polymer residue keys derived from peptide/phosphodiester bond connectivity.
    /// Used as Method D fallback when `seqres` is empty.
    /// Each entry is `(chain, seq_num, ins_code)`.
    pub polymer_residue_keys: HashSet<(char, i32, Option<char>)>,
}

/// Detect polymer residues via peptide-bond (C→N < 1.45 Å) and
/// phosphodiester-bond (O3'→P < 2.0 Å) connectivity.
///
/// Algorithm:
/// 1. Group atoms by residue key (chain, seq_num, ins_code).
/// 2. For each chain, sort residues by (seq_num, ins_code).
/// 3. For each consecutive pair, check inter-residue bond distances.
/// 4. Both residues are marked polymer if a bond is found.
pub fn compute_polymer_keys(atoms: &[Atom]) -> HashSet<(char, i32, Option<char>)> {
    use std::collections::BTreeMap;

    // residue_key → [(atom_name_trimmed, position)]
    let mut residues: HashMap<(char, i32, Option<char>), Vec<(&str, Vec3)>> = HashMap::new();
    for atom in atoms {
        let key = (atom.residue.chain, atom.residue.seq_num, atom.residue.ins_code);
        residues
            .entry(key)
            .or_default()
            .push((atom.name_trimmed(), atom.position));
    }

    // Group residue keys by chain; sort per chain by (seq_num, ins_code)
    let mut by_chain: BTreeMap<char, Vec<(char, i32, Option<char>)>> = BTreeMap::new();
    for &key in residues.keys() {
        by_chain.entry(key.0).or_default().push(key);
    }
    for keys in by_chain.values_mut() {
        keys.sort_by_key(|k| (k.1, k.2));
    }

    let mut polymer = HashSet::new();

    for keys in by_chain.values() {
        for window in keys.windows(2) {
            let key_i = window[0];
            let key_j = window[1];

            let atoms_i = &residues[&key_i];
            let atoms_j = &residues[&key_j];

            // Helper: find position of named atom in a residue's atom list
            let find_pos = |list: &[(&str, Vec3)], name: &str| -> Option<Vec3> {
                list.iter().find(|(n, _)| *n == name).map(|(_, p)| *p)
            };

            let mut bonded = false;

            // Peptide bond: C(i) → N(j) < 1.45 Å
            if let (Some(c_pos), Some(n_pos)) =
                (find_pos(atoms_i, "C"), find_pos(atoms_j, "N"))
            {
                if c_pos.distance(n_pos) < 1.45 {
                    bonded = true;
                }
            }

            // Phosphodiester bond: O3'(i) → P(j) < 2.0 Å
            if !bonded {
                if let (Some(o3_pos), Some(p_pos)) =
                    (find_pos(atoms_i, "O3'"), find_pos(atoms_j, "P"))
                {
                    if o3_pos.distance(p_pos) < 2.0 {
                        bonded = true;
                    }
                }
            }

            if bonded {
                polymer.insert(key_i);
                polymer.insert(key_j);
            }
        }
    }

    polymer
}

impl Structure {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build chain_ranges index after atoms are populated
    pub fn build_index(&mut self) {
        self.chain_ranges.clear();
        if self.atoms.is_empty() {
            return;
        }
        let mut current_chain = self.atoms[0].residue.chain;
        let mut start = 0usize;
        for (i, atom) in self.atoms.iter().enumerate() {
            if atom.residue.chain != current_chain {
                self.chain_ranges.insert(current_chain, start..i);
                current_chain = atom.residue.chain;
                start = i;
            }
        }
        self.chain_ranges.insert(current_chain, start..self.atoms.len());
    }

    /// Returns true when `atom` belongs to a polymer chain (protein or nucleic acid)
    /// rather than being a free ligand, ion, or solvent.
    ///
    /// Classification (Method F = C with D fallback):
    /// - ATOM record → always polymer.
    /// - HETATM + SEQRES present (Method C): polymer iff the residue name appears
    ///   in the SEQRES list for its chain (correctly handles MSE, amino-acid
    ///   ligands in a different chain, etc.).
    /// - HETATM + SEQRES absent (Method D fallback): polymer iff the residue's
    ///   key appears in `polymer_residue_keys`, which is built from peptide-bond
    ///   (C→N < 1.45 Å) and phosphodiester-bond (O3'→P < 2.0 Å) connectivity.
    pub fn is_polymer_atom(&self, atom: &Atom) -> bool {
        if !atom.is_hetatm {
            return true;
        }
        if !self.seqres.is_empty() {
            // Method C: SEQRES-based classification.
            return self.seqres
                .get(&atom.residue.chain)
                .map_or(false, |r| r.contains(atom.residue.name.trim()));
        }
        // Method D fallback: connectivity-based classification.
        let key = (atom.residue.chain, atom.residue.seq_num, atom.residue.ins_code);
        self.polymer_residue_keys.contains(&key)
    }

    /// Populate `polymer_residue_keys` by scanning peptide-bond and
    /// phosphodiester-bond connectivity between consecutive residues.
    ///
    /// Called from `parse_pdb` when SEQRES records are absent.
    pub fn build_polymer_keys_from_connectivity(&mut self) {
        self.polymer_residue_keys = compute_polymer_keys(&self.atoms);
    }

    pub fn centroid(&self) -> Vec3 {
        if self.atoms.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = self.atoms.iter().map(|a| a.position).sum();
        sum / self.atoms.len() as f32
    }
}
