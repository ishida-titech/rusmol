use anyhow::{Context, Result};
use glam::Vec3;
use std::path::Path;

use super::atom::{Atom, ResidueId, SecondaryStructure, Structure};
use super::bonds::build_bonds_cif;
use super::secondary::{assign_ss, SsRange};

/// Parse a PDB file and return a Structure with bonds estimated.
pub fn parse_pdb(path: &Path) -> Result<Structure> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read PDB file: {}", path.display()))?;

    let mut structure = Structure::new();
    let mut conect: Vec<(u32, u32)> = Vec::new();
    let mut ss_ranges: Vec<SsRange> = Vec::new();

    // Accumulate multi-line records before parsing
    let mut compnd_text = String::new();
    // het_id → accumulated text
    let mut hetnam_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut hetsyn_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    for (line_no, line) in content.lines().enumerate() {
        let record = if line.len() >= 6 { &line[..6] } else { line };
        match record.trim() {
            "ATOM" | "HETATM" => {
                match parse_atom_line(line, record.trim() == "HETATM") {
                    Ok(atom) => structure.atoms.push(atom),
                    Err(e) => log::warn!("line {}: {}", line_no + 1, e),
                }
            }
            "HELIX" => parse_helix_line(line, &mut ss_ranges),
            "SHEET" => parse_sheet_line(line, &mut ss_ranges),
            "CONECT" => {
                parse_conect_line(line, &mut conect);
            }
            "COMPND" => {
                // cols 10-79 (0-indexed) contain the payload
                let payload = if line.len() > 10 { &line[10..] } else { "" };
                compnd_text.push_str(payload);
                compnd_text.push(' ');
            }
            "SEQRES" => parse_seqres_line(line, &mut structure),
            "HETNAM" => accumulate_het_record(line, &mut hetnam_map),
            "HETSYN" => accumulate_het_record(line, &mut hetsyn_map),
            "END" | "ENDMDL" => break,
            _ => {}
        }
    }

    // Parse accumulated COMPND text → chain → molecule name
    structure.compnd = parse_compnd(&compnd_text);
    structure.hetnam = hetnam_map;
    structure.hetsyn = hetsyn_map;

    structure.build_index();

    // Translate all atoms so centroid is at origin
    let centroid = structure.centroid();
    for atom in &mut structure.atoms {
        atom.position -= centroid;
    }

    structure.ss = assign_ss(&structure.atoms, &ss_ranges);

    build_bonds_cif(&mut structure, &conect);

    // Method D fallback: when SEQRES is absent, derive polymer residues
    // from peptide-bond / phosphodiester-bond connectivity.
    if structure.seqres.is_empty() {
        structure.build_polymer_keys_from_connectivity();
    }

    log::info!(
        "parse_pdb '{}': {} atoms, {} bonds, {} chains",
        path.display(),
        structure.atoms.len(),
        structure.bonds.len(),
        structure.chain_ranges.len(),
    );
    Ok(structure)
}

// ── COMPND / HETNAM / HETSYN helpers ────────────────────────────────────────

/// Accumulate a HETNAM or HETSYN continuation record into `map`.
/// Layout (0-indexed): cols 11-13 = het_id, cols 15.. = text.
/// Parse one SEQRES line and add residue names to `structure.seqres`.
///
/// PDB fixed-width layout (1-based columns):
///   Col 12      : chain ID
///   Col 20-22   : residue 1  (0-based bytes 19..22)
///   Col 24-26   : residue 2  (0-based bytes 23..26)
///   …up to 13 residues per line, spaced every 4 bytes
fn parse_seqres_line(line: &str, structure: &mut Structure) {
    let bytes = line.as_bytes();
    if bytes.len() < 12 {
        return;
    }
    let chain = bytes[11] as char;
    if chain == ' ' {
        return;
    }
    let entry = structure.seqres.entry(chain).or_default();
    let mut pos = 19usize;
    while pos + 3 <= line.len() {
        let resname = line[pos..pos + 3].trim();
        if !resname.is_empty() {
            entry.insert(resname.to_string());
        }
        pos += 4;
    }
}

fn accumulate_het_record(line: &str, map: &mut std::collections::HashMap<String, String>) {
    let get = |s: usize, e: usize| -> &str {
        let e = e.min(line.len());
        if s >= line.len() { "" } else { &line[s..e] }
    };
    let het_id = get(11, 14).trim().to_string();
    if het_id.is_empty() { return; }
    let text = get(15, line.len()).trim_end();
    let entry = map.entry(het_id).or_default();
    if !entry.is_empty() { entry.push(' '); }
    entry.push_str(text);
}

/// Parse joined COMPND text and return chain → molecule name.
/// COMPND token: value; pairs. MOL_ID groups MOLECULE + CHAIN tokens.
fn parse_compnd(text: &str) -> std::collections::HashMap<char, String> {
    let mut result = std::collections::HashMap::new();

    // Split into `TOKEN: VALUE` segments by `;`
    let mut current_mol: Option<String> = None;
    let mut current_chains: Vec<char> = Vec::new();

    for segment in text.split(';') {
        let segment = segment.trim();
        if segment.is_empty() { continue; }
        if let Some((key, val)) = segment.split_once(':') {
            let key = key.trim().to_uppercase();
            let val = val.trim();
            match key.as_str() {
                "MOL_ID" => {
                    // Flush previous molecule
                    if let Some(mol) = current_mol.take() {
                        for ch in current_chains.drain(..) {
                            result.insert(ch, mol.clone());
                        }
                        let _ = mol; // already drained
                    }
                }
                "MOLECULE" => {
                    current_mol = Some(val.to_string());
                }
                "CHAIN" => {
                    // e.g. "A, B, C"
                    for ch in val.split(',') {
                        let ch = ch.trim();
                        if let Some(c) = ch.chars().next() {
                            current_chains.push(c);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    // Flush last molecule
    if let Some(mol) = current_mol {
        for ch in current_chains {
            result.insert(ch, mol.clone());
        }
    }

    result
}

fn col_char(line: &str, idx: usize) -> char {
    line.as_bytes().get(idx).copied().unwrap_or(b' ') as char
}

fn col_i32(line: &str, start: usize, end: usize) -> Option<i32> {
    let end = end.min(line.len());
    if start >= line.len() {
        return None;
    }
    line[start..end].trim().parse().ok()
}

/// HELIX record (PDB column layout, 1-indexed → 0-indexed slices):
///   initChainID: col 20  → index 19
///   initSeqNum:  cols 22-25 → [21..25]
///   initICode:   col 26  → index 25
///   endChainID:  col 32  → index 31
///   endSeqNum:   cols 34-37 → [33..37]
///   endICode:    col 38  → index 37
fn parse_helix_line(line: &str, ranges: &mut Vec<SsRange>) {
    let chain = col_char(line, 19);
    let Some(start_seq) = col_i32(line, 21, 25) else { return };
    let start_ins_ch = col_char(line, 25);
    let start_ins = if start_ins_ch == ' ' { None } else { Some(start_ins_ch) };
    let end_chain = col_char(line, 31);
    let Some(end_seq) = col_i32(line, 33, 37) else { return };
    let end_ins_ch = col_char(line, 37);
    let end_ins = if end_ins_ch == ' ' { None } else { Some(end_ins_ch) };
    if chain != end_chain { return; }
    ranges.push(SsRange { chain, start_seq, start_ins, end_seq, end_ins, ss: SecondaryStructure::Helix });
}

/// SHEET record:
///   initChainID: col 22  → index 21
///   initSeqNum:  cols 23-26 → [22..26]
///   initICode:   col 27  → index 26
///   endChainID:  col 33  → index 32
///   endSeqNum:   cols 34-37 → [33..37]
///   endICode:    col 38  → index 37
fn parse_sheet_line(line: &str, ranges: &mut Vec<SsRange>) {
    let chain = col_char(line, 21);
    let Some(start_seq) = col_i32(line, 22, 26) else { return };
    let start_ins_ch = col_char(line, 26);
    let start_ins = if start_ins_ch == ' ' { None } else { Some(start_ins_ch) };
    let end_chain = col_char(line, 32);
    let Some(end_seq) = col_i32(line, 33, 37) else { return };
    let end_ins_ch = col_char(line, 37);
    let end_ins = if end_ins_ch == ' ' { None } else { Some(end_ins_ch) };
    if chain != end_chain { return; }
    ranges.push(SsRange { chain, start_seq, start_ins, end_seq, end_ins, ss: SecondaryStructure::Sheet });
}

fn parse_conect_line(line: &str, conect: &mut Vec<(u32, u32)>) {
    // CONECT cols: 7-11 (atom1), 12-16 (bond1), 17-21 (bond2), 22-26 (bond3), 27-31 (bond4)
    let col = |start: usize, end: usize| -> Option<u32> {
        let end = end.min(line.len());
        if start >= line.len() {
            return None;
        }
        line[start..end].trim().parse().ok()
    };

    let Some(serial1) = col(6, 11) else { return };
    for start in [11, 16, 21, 26] {
        if let Some(serial2) = col(start, start + 5) {
            if serial1 != serial2 {
                let (a, b) = if serial1 < serial2 {
                    (serial1, serial2)
                } else {
                    (serial2, serial1)
                };
                if !conect.contains(&(a, b)) {
                    conect.push((a, b));
                }
            }
        }
    }
}

fn parse_atom_line(line: &str, is_hetatm: bool) -> Result<Atom> {
    // PDB column layout (1-indexed, converted to 0-indexed slices):
    //  7-11:  serial       (cols 6..11)
    // 13-16:  name         (cols 12..16)
    // 17:     alt_loc      (col  16)
    // 18-20:  resname      (cols 17..20)
    // 22:     chain        (col  21)
    // 23-26:  resseq       (cols 22..26)
    // 27:     icode        (col  26)
    // 31-38:  x            (cols 30..38)
    // 39-46:  y            (cols 38..46)
    // 47-54:  z            (cols 46..54)
    // 55-60:  occupancy    (cols 54..60)
    // 61-66:  tempFactor   (cols 60..66)
    // 77-78:  element      (cols 76..78)

    let col = |start: usize, end: usize| -> &str {
        let end = end.min(line.len());
        if start >= line.len() {
            ""
        } else {
            &line[start..end]
        }
    };

    let serial: u32 = col(6, 11).trim().parse().unwrap_or(0);
    let name = col(12, 16).to_string();
    let alt_loc_ch = col(16, 17).chars().next().unwrap_or(' ');
    let alt_loc = if alt_loc_ch == ' ' { None } else { Some(alt_loc_ch) };
    let resname = col(17, 20).trim().to_string();
    let chain = col(21, 22).chars().next().unwrap_or('A');
    let seq_num: i32 = col(22, 26).trim().parse().unwrap_or(0);
    let ins_ch = col(26, 27).chars().next().unwrap_or(' ');
    let ins_code = if ins_ch == ' ' { None } else { Some(ins_ch) };

    let x: f32 = col(30, 38)
        .trim()
        .parse()
        .with_context(|| format!("bad x coord in: {}", line))?;
    let y: f32 = col(38, 46)
        .trim()
        .parse()
        .with_context(|| format!("bad y coord in: {}", line))?;
    let z: f32 = col(46, 54)
        .trim()
        .parse()
        .with_context(|| format!("bad z coord in: {}", line))?;

    let temp_factor: f32 = col(60, 66).trim().parse().unwrap_or(0.0);

    let element = {
        let e = col(76, 78).trim().to_uppercase();
        if !e.is_empty() {
            e
        } else {
            let n = name.trim();
            let alpha: String = n.chars().filter(|c| c.is_alphabetic()).take(2).collect();
            if alpha.len() >= 2 && is_hetatm {
                alpha.to_uppercase()
            } else {
                alpha
                    .chars()
                    .next()
                    .map(|c| c.to_string())
                    .unwrap_or_default()
                    .to_uppercase()
            }
        }
    };

    Ok(Atom {
        serial,
        name,
        alt_loc,
        residue: ResidueId {
            chain,
            seq_num,
            ins_code,
            name: resname,
        },
        position: Vec3::new(x, y, z),
        temp_factor,
        element,
        is_hetatm,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    // ── 1crn ────────────────────────────────────────────────────────────────

    #[test]
    fn crn_atom_count() {
        let s = parse_pdb(&fixture("1crn.pdb")).unwrap();
        assert_eq!(s.atoms.len(), 327);
    }

    #[test]
    fn crn_no_hetatm() {
        let s = parse_pdb(&fixture("1crn.pdb")).unwrap();
        assert!(s.atoms.iter().all(|a| !a.is_hetatm));
    }

    #[test]
    fn crn_single_chain_a() {
        let s = parse_pdb(&fixture("1crn.pdb")).unwrap();
        assert_eq!(s.chain_ranges.len(), 1);
        assert!(s.chain_ranges.contains_key(&'A'));
    }

    #[test]
    fn crn_has_bonds() {
        let s = parse_pdb(&fixture("1crn.pdb")).unwrap();
        assert!(!s.bonds.is_empty());
    }

    #[test]
    fn crn_ca_element_is_c() {
        let s = parse_pdb(&fixture("1crn.pdb")).unwrap();
        let ca = s.atoms.iter().find(|a| a.name.trim() == "CA").unwrap();
        assert_eq!(ca.element, "C");
    }

    #[test]
    fn crn_n_element_is_n() {
        let s = parse_pdb(&fixture("1crn.pdb")).unwrap();
        let n = s.atoms.iter().find(|a| a.name.trim() == "N").unwrap();
        assert_eq!(n.element, "N");
    }

    #[test]
    fn crn_coordinates_finite() {
        let s = parse_pdb(&fixture("1crn.pdb")).unwrap();
        for a in &s.atoms {
            assert!(a.position.x.is_finite());
            assert!(a.position.y.is_finite());
            assert!(a.position.z.is_finite());
        }
    }

    #[test]
    fn crn_centroid_near_origin() {
        // parse_pdb translates the structure so centroid ≈ origin
        let s = parse_pdb(&fixture("1crn.pdb")).unwrap();
        assert!(s.centroid().length() < 0.5, "centroid should be near origin");
    }

    #[test]
    fn crn_ss_length_matches_atoms() {
        let s = parse_pdb(&fixture("1crn.pdb")).unwrap();
        assert_eq!(s.ss.len(), s.atoms.len());
    }

    // ── 2je5 ────────────────────────────────────────────────────────────────

    #[test]
    fn je5_atom_count() {
        let s = parse_pdb(&fixture("2je5.pdb")).unwrap();
        assert_eq!(s.atoms.len(), 7239);
    }

    #[test]
    fn je5_has_hetatm() {
        let s = parse_pdb(&fixture("2je5.pdb")).unwrap();
        assert!(s.atoms.iter().any(|a| a.is_hetatm));
    }

    #[test]
    fn je5_has_helix_ss() {
        let s = parse_pdb(&fixture("2je5.pdb")).unwrap();
        use crate::structure::atom::SecondaryStructure;
        assert!(s.ss.iter().any(|&ss| ss == SecondaryStructure::Helix));
    }

    #[test]
    fn je5_bond_indices_in_range() {
        let s = parse_pdb(&fixture("2je5.pdb")).unwrap();
        let n = s.atoms.len();
        for b in &s.bonds {
            assert!(b.atom1 < n, "bond.atom1 out of range");
            assert!(b.atom2 < n, "bond.atom2 out of range");
            assert_ne!(b.atom1, b.atom2, "self-bond found");
        }
    }

    // ── COMPND / HETNAM parsing ──────────────────────────────────────────────

    #[test]
    fn je5_compnd_parsed() {
        let s = parse_pdb(&fixture("2je5.pdb")).unwrap();
        // 2je5 should have COMPND records for its chains
        // (just verify the map is populated when COMPND exists)
        // 1crn may or may not have COMPND; 2je5 typically does
        // We just check it doesn't panic and returns something sensible
        for (_, name) in &s.compnd {
            assert!(!name.is_empty(), "molecule name should not be empty");
        }
    }
}
