use std::fs;
use std::path::PathBuf;

// ── Built-in bond tables ───────────────────────────────────────────────────────
// Heavy-atom (non-H) covalent bonds for the 20 standard amino acids,
// common modified residues, and frequent small molecules.
// Sourced from RCSB CCD; included here to avoid network access for common inputs.

pub fn builtin_bonds(resname: &str) -> Option<&'static [(&'static str, &'static str)]> {
    match resname {
        "ALA" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),
        ]),
        "ARG" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","CD"),("CD","NE"),
            ("NE","CZ"),("CZ","NH1"),("CZ","NH2"),
        ]),
        "ASN" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","OD1"),("CG","ND2"),
        ]),
        "ASP" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","OD1"),("CG","OD2"),
        ]),
        "CYS" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","SG"),
        ]),
        "GLN" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","CD"),("CD","OE1"),("CD","NE2"),
        ]),
        "GLU" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","CD"),("CD","OE1"),("CD","OE2"),
        ]),
        "GLY" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
        ]),
        "HIS" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","ND1"),("CG","CD2"),
            ("ND1","CE1"),("CE1","NE2"),("NE2","CD2"),
        ]),
        "ILE" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG1"),("CB","CG2"),("CG1","CD1"),
        ]),
        "LEU" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","CD1"),("CG","CD2"),
        ]),
        "LYS" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","CD"),("CD","CE"),("CE","NZ"),
        ]),
        "MET" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","SD"),("SD","CE"),
        ]),
        "PHE" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","CD1"),("CG","CD2"),
            ("CD1","CE1"),("CD2","CE2"),("CE1","CZ"),("CE2","CZ"),
        ]),
        "PRO" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","CD"),("CD","N"),
        ]),
        "SER" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","OG"),
        ]),
        "THR" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","OG1"),("CB","CG2"),
        ]),
        "TRP" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),
            ("CG","CD1"),("CG","CD2"),
            ("CD1","NE1"),("NE1","CE2"),("CE2","CD2"),
            ("CE2","CZ2"),("CZ2","CH2"),("CH2","CZ3"),("CZ3","CE3"),("CE3","CD2"),
        ]),
        "TYR" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","CD1"),("CG","CD2"),
            ("CD1","CE1"),("CD2","CE2"),("CE1","CZ"),("CE2","CZ"),("CZ","OH"),
        ]),
        "VAL" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG1"),("CB","CG2"),
        ]),

        // ── Modified / non-standard ──────────────────────────────────────────
        // MSE: selenomethionine (SE replaces S)
        "MSE" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","SE"),("SE","CE"),
        ]),
        // SEC: selenocysteine
        "SEC" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","SEG"),
        ]),
        // HYP: trans-4-hydroxyproline
        "HYP" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","CD"),("CD","N"),("CG","OD1"),
        ]),
        // CME: S,S-(2-hydroxyethyl)thiocysteine (cysteine adduct)
        "CME" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","SG"),("SG","C1"),("C1","C2"),("C2","O2"),
        ]),
        // MLY: N-dimethyl-lysine
        "MLY" => Some(&[
            ("N","CA"),("CA","C"),("C","O"),("C","OXT"),
            ("CA","CB"),("CB","CG"),("CG","CD"),("CD","CE"),
            ("CE","NZ"),("NZ","C1"),("NZ","C2"),
        ]),

        // ── Common small molecules ───────────────────────────────────────────
        "HOH" | "WAT" | "DOD" => Some(&[]),
        "SO4" => Some(&[("S","O1"),("S","O2"),("S","O3"),("S","O4")]),
        "PO4" => Some(&[("P","O1"),("P","O2"),("P","O3"),("P","O4")]),
        "GOL" => Some(&[ // glycerol
            ("C1","O1"),("C1","C2"),("C2","O2"),("C2","C3"),("C3","O3"),
        ]),
        "EDO" => Some(&[ // ethylene glycol
            ("C1","O1"),("C1","C2"),("C2","O2"),
        ]),
        "ACT" | "ACY" => Some(&[ // acetate / acetic acid
            ("C","OXT"),("C","O"),("C","CH3"),
        ]),
        "CL" | "BR" | "IOD" | "ZN" | "MG" | "CA" | "NA" | "K" |
        "FE" | "CU" | "MN" | "CO" | "NI" => Some(&[]), // monoatomic ions

        _ => None,
    }
}

// ── Network / cache access ─────────────────────────────────────────────────────

/// Return covalent bond pairs for a residue.
/// Checks built-in table first; downloads from RCSB CCD only for unknown residues.
pub fn fetch_component_bonds(resname: &str) -> Vec<(String, String)> {
    if let Some(pairs) = builtin_bonds(resname) {
        return pairs.iter().map(|&(a, b)| (a.to_string(), b.to_string())).collect();
    }

    let cache_path = cache_file(resname);

    let content = if cache_path.exists() {
        match fs::read_to_string(&cache_path) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("CCD cache read failed for '{resname}': {e}");
                return vec![];
            }
        }
    } else {
        eprintln!("Downloading CCD for '{resname}'...");
        match download_cif(resname) {
            Ok(c) => {
                if let Some(parent) = cache_path.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                let _ = fs::write(&cache_path, &c);
                c
            }
            Err(e) => {
                log::warn!("CCD download failed for '{resname}': {e}");
                return vec![];
            }
        }
    };

    parse_bonds_from_cif(&content)
}

fn download_cif(resname: &str) -> anyhow::Result<String> {
    let url = format!(
        "https://files.rcsb.org/ligands/download/{}.cif",
        resname.to_uppercase()
    );
    let body = ureq::get(&url)
        .call()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .into_string()?;
    Ok(body)
}

fn cache_file(resname: &str) -> PathBuf {
    let base = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join(".cache")
        .join("rusmol")
        .join("ccd")
        .join(format!("{}.cif", resname.to_uppercase()))
}

// ── CIF parser ─────────────────────────────────────────────────────────────────

/// Parse `_chem_comp_bond` loop from CIF content.
/// Returns (atom_id_1, atom_id_2) for every listed bond.
pub fn parse_bonds_from_cif(content: &str) -> Vec<(String, String)> {
    let mut bonds = Vec::new();
    let mut in_bond_loop = false;
    let mut fields: Vec<&str> = Vec::new();
    let mut atom1_col: Option<usize> = None;
    let mut atom2_col: Option<usize> = None;

    for raw in content.lines() {
        let line = raw.trim();

        if line == "loop_" {
            in_bond_loop = false;
            fields.clear();
            atom1_col = None;
            atom2_col = None;
            continue;
        }

        if line.starts_with("_chem_comp_bond.") {
            in_bond_loop = true;
            let col = fields.len();
            if line == "_chem_comp_bond.atom_id_1" {
                atom1_col = Some(col);
            } else if line == "_chem_comp_bond.atom_id_2" {
                atom2_col = Some(col);
            }
            fields.push(line);
            continue;
        }

        if line.starts_with('_') || line.starts_with('#') || line.is_empty() {
            in_bond_loop = false;
            continue;
        }

        if in_bond_loop {
            if let (Some(c1), Some(c2)) = (atom1_col, atom2_col) {
                let tokens = tokenize(line);
                if tokens.len() > c1.max(c2) {
                    bonds.push((tokens[c1].clone(), tokens[c2].clone()));
                }
            }
        }
    }

    bonds
}

fn tokenize(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        if bytes[i] == b'\'' || bytes[i] == b'"' {
            let q = bytes[i];
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != q {
                i += 1;
            }
            tokens.push(line[start..i].to_string());
            if i < bytes.len() {
                i += 1;
            }
        } else {
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            tokens.push(line[start..i].to_string());
        }
    }

    tokens
}
