use std::collections::HashMap;

use super::atom::{Bond, Structure};
use super::ccd;

/// Build bonds using RCSB CCD for intraresidue connectivity,
/// plus topology-based checks for interpeptide/disulfide/nucleotide bonds.
///
/// For residues not in the built-in table, CONECT records are checked first.
/// CIF is downloaded only when a residue has no CONECT coverage.
pub fn build_bonds_cif(structure: &mut Structure, conect: &[(u32, u32)]) {
    let resnames: std::collections::HashSet<String> =
        structure.atoms.iter().map(|a| a.residue.name.clone()).collect();

    // Set of atom serials that appear in at least one CONECT record
    let conect_serials: std::collections::HashSet<u32> = conect
        .iter()
        .flat_map(|&(s1, s2)| [s1, s2])
        .collect();

    // Build CCD map: builtin → instant, unknown → CONECT first, then CIF
    let mut ccd_bonds: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for resname in &resnames {
        if ccd::builtin_bonds(resname).is_some() {
            // Fast path: no allocation beyond the static slice
            let pairs = ccd::fetch_component_bonds(resname);
            ccd_bonds.insert(resname.clone(), pairs);
        } else {
            // Check whether any atom of this residue has CONECT records
            let has_conect = structure
                .atoms
                .iter()
                .filter(|a| &a.residue.name == resname)
                .any(|a| conect_serials.contains(&a.serial));

            if has_conect {
                log::debug!("'{resname}': skipping CCD download (CONECT records present)");
            } else {
                let pairs = ccd::fetch_component_bonds(resname);
                if !pairs.is_empty() {
                    ccd_bonds.insert(resname.clone(), pairs);
                }
            }
        }
    }

    // Group atom indices by residue
    type ResKey = (char, i32, Option<char>, String);
    let mut residue_atoms: HashMap<ResKey, Vec<usize>> = HashMap::new();
    for (i, atom) in structure.atoms.iter().enumerate() {
        let key = (
            atom.residue.chain,
            atom.residue.seq_num,
            atom.residue.ins_code,
            atom.residue.name.clone(),
        );
        residue_atoms.entry(key).or_default().push(i);
    }

    let mut bonds: Vec<Bond> = Vec::new();

    // Intraresidue bonds from CCD
    for (res_key, atom_indices) in &residue_atoms {
        let Some(bond_defs) = ccd_bonds.get(&res_key.3) else { continue };

        let name_map: HashMap<&str, usize> = atom_indices
            .iter()
            .map(|&i| (structure.atoms[i].name.trim(), i))
            .collect();

        for (n1, n2) in bond_defs {
            if let (Some(&i), Some(&j)) = (name_map.get(n1.as_str()), name_map.get(n2.as_str())) {
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                bonds.push(Bond { atom1: lo, atom2: hi });
            }
        }
    }

    // Inter-residue covalent bonds
    add_inter_residue_bonds(structure, &mut bonds);

    structure.bonds = bonds;

    // CONECT records: always applied last so they can supplement CIF or builtin bonds
    add_conect_bonds(structure, conect);

    // Hydrogens are absent from the built-in / CCD bond tables, so any H present in
    // the input (e.g. PDBQT polar hydrogens) would otherwise render as a floating
    // sphere. Connect each still-bondless H to its nearest heavy atom by distance.
    add_hydrogen_bonds(structure);

    log::debug!(
        "Built {} bonds for {} atoms",
        structure.bonds.len(),
        structure.atoms.len()
    );
}

// ── Inter-residue bonds ────────────────────────────────────────────────────────

/// Add covalent bonds that span residue boundaries:
/// - Peptide bonds (C→N): determined by consecutive residue numbers.
///   Distance is validated and a warning is emitted if abnormal.
/// - Disulfide bonds (CYS SG–SG): distance < 2.30 Å.
/// - Nucleotide backbone (O3'→P): distance < 1.70 Å.
fn add_inter_residue_bonds(structure: &Structure, bonds: &mut Vec<Bond>) {
    type ResId = (char, i32, Option<char>); // (chain, seq_num, ins_code)

    let mut c_atoms: HashMap<ResId, (usize, glam::Vec3)> = HashMap::new();
    let mut n_atoms: HashMap<ResId, (usize, glam::Vec3)> = HashMap::new();
    let mut sg_atoms: Vec<(usize, glam::Vec3)> = Vec::new();
    let mut o3_atoms: Vec<(usize, glam::Vec3)> = Vec::new();
    let mut p_atoms: Vec<(usize, glam::Vec3)> = Vec::new();

    // Unique (seq_num, ins_code) per chain
    let mut chain_residues: HashMap<char, Vec<(i32, Option<char>)>> = HashMap::new();

    for (i, atom) in structure.atoms.iter().enumerate() {
        let chain = atom.residue.chain;
        let res_id: ResId = (chain, atom.residue.seq_num, atom.residue.ins_code);

        chain_residues
            .entry(chain)
            .or_default()
            .push((atom.residue.seq_num, atom.residue.ins_code));

        match atom.name.trim() {
            "C"   => { c_atoms.entry(res_id).or_insert((i, atom.position)); }
            "N"   => { n_atoms.entry(res_id).or_insert((i, atom.position)); }
            "SG" if matches!(atom.residue.name.as_str(), "CYS" | "CYX") => {
                sg_atoms.push((i, atom.position));
            }
            "O3'" => o3_atoms.push((i, atom.position)),
            "P"   => p_atoms.push((i, atom.position)),
            _ => {}
        }
    }

    // Deduplicate and sort residue lists per chain
    for res_list in chain_residues.values_mut() {
        res_list.sort_unstable();
        res_list.dedup();
    }

    // Peptide bonds by consecutive residue numbers
    for (chain, res_list) in &chain_residues {
        for window in res_list.windows(2) {
            let (seq1, ins1) = window[0];
            let (seq2, ins2) = window[1];

            if !is_consecutive(seq1, ins1, seq2, ins2) {
                continue;
            }

            let id1: ResId = (*chain, seq1, ins1);
            let id2: ResId = (*chain, seq2, ins2);

            let Some(&(ic, pc)) = c_atoms.get(&id1) else { continue };
            let Some(&(in_, pn)) = n_atoms.get(&id2) else { continue };

            let dist = (pc - pn).length();

            // Typical peptide C-N bond: 1.28–1.40 Å.
            // Warn if outside a generous tolerance.
            if !(1.1..=2.0).contains(&dist) {
                log::warn!(
                    "unusual C-N distance {:.2} Å between residues {}{} and {}{} (chain {})",
                    dist,
                    seq1, ins1.map(|c| c.to_string()).unwrap_or_default(),
                    seq2, ins2.map(|c| c.to_string()).unwrap_or_default(),
                    chain
                );
            }

            let (lo, hi) = if ic < in_ { (ic, in_) } else { (in_, ic) };
            if !bonds.iter().any(|e| e.atom1 == lo && e.atom2 == hi) {
                bonds.push(Bond { atom1: lo, atom2: hi });
            }
        }
    }

    // Disulfide bonds (distance-based between CYS SG atoms)
    for i in 0..sg_atoms.len() {
        for j in (i + 1)..sg_atoms.len() {
            let (ia, pa) = sg_atoms[i];
            let (ib, pb) = sg_atoms[j];
            if (pa - pb).length() < 2.30 {
                bonds.push(Bond { atom1: ia.min(ib), atom2: ia.max(ib) });
            }
        }
    }

    // Nucleotide backbone O3'→P (distance-based)
    for &(io, po) in &o3_atoms {
        for &(ip, pp) in &p_atoms {
            if (po - pp).length() < 1.70 {
                let (lo, hi) = if io < ip { (io, ip) } else { (ip, io) };
                bonds.push(Bond { atom1: lo, atom2: hi });
            }
        }
    }
}

/// True when residue (seq2, ins2) immediately follows (seq1, ins1) in sequence.
///
/// Handles:
/// - Normal:          14  → 15   (no ins_code)
/// - Insertion start: 14  → 14A
/// - Insertion cont:  14A → 14B
/// - Insertion end:   14B → 15
fn is_consecutive(seq1: i32, ins1: Option<char>, seq2: i32, ins2: Option<char>) -> bool {
    let i1 = ins1.unwrap_or(' ');
    let i2 = ins2.unwrap_or(' ');

    if i1 == ' ' && i2 == ' ' {
        // Plain consecutive numbers
        seq2 == seq1 + 1
    } else if seq1 == seq2 {
        // Same seq_num: ' '→'A', 'A'→'B', …
        let next = if i1 == ' ' { 'A' } else { (i1 as u8 + 1) as char };
        i2 == next
    } else if seq2 == seq1 + 1 {
        // Last insertion-code variant to next clean number: 14B → 15
        i1 != ' ' && i2 == ' '
    } else {
        false
    }
}

// ── Hydrogen bonds (connectivity, not H-bonds) ──────────────────────────────────

/// Connect hydrogens to their nearest heavy atom. The built-in and CCD bond
/// tables only cover heavy atoms, so hydrogens present in the input arrive with
/// no connectivity. Only hydrogens that are still bondless are considered (so
/// explicit CONECT/CCD hydrogen bonds are preserved), and the search is limited
/// to the same residue — where a covalent X–H partner always lives.
fn add_hydrogen_bonds(structure: &mut Structure) {
    use std::collections::HashSet;
    /// Upper bound for an X–H covalent bond length (Å). Real X–H is ~0.9–1.1 Å.
    const MAX_XH: f32 = 1.3;

    let mut bonded: HashSet<usize> = HashSet::new();
    for b in &structure.bonds {
        bonded.insert(b.atom1);
        bonded.insert(b.atom2);
    }

    // Group atom indices by residue so the nearest-heavy-atom search stays local.
    type ResKey = (char, i32, Option<char>, String);
    let mut residue_atoms: HashMap<ResKey, Vec<usize>> = HashMap::new();
    for (i, atom) in structure.atoms.iter().enumerate() {
        let key = (
            atom.residue.chain,
            atom.residue.seq_num,
            atom.residue.ins_code,
            atom.residue.name.clone(),
        );
        residue_atoms.entry(key).or_default().push(i);
    }

    let mut new_bonds: Vec<Bond> = Vec::new();
    for indices in residue_atoms.values() {
        for &i in indices {
            let atom = &structure.atoms[i];
            if atom.element != "H" || bonded.contains(&i) {
                continue;
            }
            let mut best: Option<(usize, f32)> = None;
            for &j in indices {
                if j == i || structure.atoms[j].element == "H" {
                    continue;
                }
                let d = (atom.position - structure.atoms[j].position).length();
                if d <= MAX_XH && best.map_or(true, |(_, bd)| d < bd) {
                    best = Some((j, d));
                }
            }
            if let Some((j, _)) = best {
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                new_bonds.push(Bond { atom1: lo, atom2: hi });
            }
        }
    }
    structure.bonds.extend(new_bonds);
}

// ── CONECT bonds ───────────────────────────────────────────────────────────────

/// Add explicit bonds from CONECT records (PDB serial numbers → atom indices).
pub fn add_conect_bonds(structure: &mut Structure, conect: &[(u32, u32)]) {
    let serial_to_idx: HashMap<u32, usize> = structure
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (a.serial, i))
        .collect();

    for &(s1, s2) in conect {
        let (Some(&i), Some(&j)) = (serial_to_idx.get(&s1), serial_to_idx.get(&s2)) else {
            continue;
        };
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        if !structure.bonds.iter().any(|e| e.atom1 == lo && e.atom2 == hi) {
            structure.bonds.push(Bond { atom1: lo, atom2: hi });
        }
    }
}
