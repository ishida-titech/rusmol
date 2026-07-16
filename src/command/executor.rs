use crate::render::camera::Camera;
use crate::scene::{Scene, SceneDirty};
use crate::structure::pdb::{parse_pdb, parse_pdbqt};
use crate::util::color::{chain_color, cpk_color, ss_color};

use super::selection::matches as sel_matches;
use super::{ColorSpec, Command, CommandResponse, SelectionExpr};
use crate::scene::object::{MolecularObject, RepresentationType};

/// Execute a command, mutating scene and/or camera.
/// Returns (response, dirty_flags) indicating which GPU data parts need re-upload.
pub fn execute(cmd: Command, scene: &mut Scene, camera: &mut Camera) -> (CommandResponse, SceneDirty) {
    match cmd {
        Command::Load { path, name } => {
            let obj_name = name.unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mol")
                    .to_string()
            });
            let is_pdbqt = path.extension().map_or(false, |e| e.eq_ignore_ascii_case("pdbqt"));
            let parse_result = if is_pdbqt { parse_pdbqt(&path) } else { parse_pdb(&path) };
            match parse_result {
                Ok(structure) => {
                    let n_atoms = structure.atoms.len();
                    let n_bonds = structure.bonds.len();
                    let summary = structure_summary(&structure);

                    // Build protein / ligand atom lists before structure is moved.
                    // is_polymer_atom() uses SEQRES when available, so modified amino
                    // acids (MSE etc.) land in protein rather than ligand.
                    const WATERS: &[&str] = &["HOH", "WAT", "DOD"];
                    let protein_sel: Vec<(String, usize)> = structure
                        .atoms.iter().enumerate()
                        .filter(|(_, a)| {
                            !WATERS.contains(&a.residue.name.trim())
                                && structure.is_polymer_atom(a)
                        })
                        .map(|(i, _)| (obj_name.clone(), i))
                        .collect();
                    let ligand_sel: Vec<(String, usize)> = structure
                        .atoms.iter().enumerate()
                        .filter(|(_, a)| {
                            !WATERS.contains(&a.residue.name.trim())
                                && !structure.is_polymer_atom(a)
                        })
                        .map(|(i, _)| (obj_name.clone(), i))
                        .collect();

                    let obj = MolecularObject::new(obj_name.clone(), structure);
                    scene.add_object(obj);

                    // Extend the global "protein" / "ligand" selections.
                    // Using extend so multiple loaded files accumulate correctly.
                    scene.selections.entry("protein".into()).or_default().extend(protein_sel);
                    scene.selections.entry("ligand".into()).or_default().extend(ligand_sel);

                    let msg = format!(
                        "Loaded '{obj_name}': {n_atoms} atoms, {n_bonds} bonds\n{summary}"
                    );
                    (CommandResponse::Ok(msg), SceneDirty::ALL)
                }
                Err(e) => (CommandResponse::Error(e.to_string()), SceneDirty::NONE),
            }
        }

        Command::Select { name, expr } => {
            let atoms = collect_matching(scene, &expr);
            let count = atoms.len();
            scene.selections.insert(name.clone(), atoms);
            (CommandResponse::Ok(format!("{count} atoms → '{name}'")), SceneDirty::NONE)
        }

        Command::Show { repr, sel } => {
            let atoms = atoms_for_sel(scene, sel.as_deref());
            if atoms.is_empty() {
                return (CommandResponse::Error(
                    sel.map(|s| format!("no atoms match '{s}'"))
                        .unwrap_or_else(|| "no atoms in scene".into()),
                ), SceneDirty::NONE);
            }
            let bit = repr.to_bit();
            // When no selector is given, skip water molecules — they are hidden by
            // default and their red CPK oxygen color would corrupt surface colors.
            let skip_water = sel.is_none();
            const WATERS: &[&str] = &["HOH", "WAT", "DOD"];
            let mut n = 0usize;
            for (obj_name, atom_idx) in &atoms {
                if let Some(obj) = scene.get_mut(obj_name) {
                    if skip_water {
                        let rn = obj.structure.atoms[*atom_idx].residue.name.trim();
                        if WATERS.contains(&rn) { continue; }
                    }
                    if let Some(f) = obj.atom_rep_show.get_mut(*atom_idx) {
                        *f |= bit;
                        n += 1;
                    }
                }
            }
            (CommandResponse::Ok(format!("show {repr:?} on {n} atoms")), dirty_for_repr(repr))
        }

        Command::Hide { repr, sel } => {
            let atoms = atoms_for_sel(scene, sel.as_deref());
            if atoms.is_empty() {
                return (CommandResponse::Error(
                    sel.map(|s| format!("no atoms match '{s}'"))
                        .unwrap_or_else(|| "no atoms in scene".into()),
                ), SceneDirty::NONE);
            }
            let bit = repr.to_bit();
            let n = atoms.len();
            for (obj_name, atom_idx) in &atoms {
                if let Some(obj) = scene.get_mut(obj_name) {
                    if let Some(f) = obj.atom_rep_show.get_mut(*atom_idx) {
                        *f &= !bit;
                    }
                }
            }
            (CommandResponse::Ok(format!("hide {repr:?} on {n} atoms")), dirty_for_repr(repr))
        }

        Command::Color { spec, sel } => {
            let atoms = match sel.as_deref() {
                Some(s) => {
                    if scene.selections.contains_key(s) {
                        scene.resolve_selection(s)
                    } else {
                        match crate::command::selection::parse(s) {
                            Ok(expr) => collect_matching(scene, &expr),
                            Err(e) => return (CommandResponse::Error(e), SceneDirty::NONE),
                        }
                    }
                }
                None => scene.all_atoms(),
            };

            let count = atoms.len();
            if count == 0 {
                return (
                    CommandResponse::Error(
                        sel.map(|s| format!("no atoms match '{s}'"))
                            .unwrap_or_else(|| "no atoms in scene".into()),
                    ),
                    SceneDirty::NONE,
                );
            }

            match &spec {
                ColorSpec::Spectrum => {
                    apply_spectrum(scene, &atoms);
                }
                ColorSpec::BFactor => {
                    apply_bfactor(scene, &atoms);
                }
                _ => {
                    // Per-atom coloring: build chain index map for Chain variant
                    let mut chain_index: std::collections::HashMap<char, usize> =
                        std::collections::HashMap::new();
                    let mut next_chain_idx = 0usize;

                    for (obj_name, atom_idx) in &atoms {
                        if let Some(obj) = scene.get_mut(obj_name) {
                            if atom_idx >= &obj.atom_colors.len() {
                                continue;
                            }
                            let color = match &spec {
                                ColorSpec::Rgb(rgb) => *rgb,
                                ColorSpec::Element => {
                                    cpk_color(&obj.structure.atoms[*atom_idx].element)
                                }
                                ColorSpec::Chain => {
                                    let ch = obj.structure.atoms[*atom_idx].residue.chain;
                                    let idx = chain_index.entry(ch).or_insert_with(|| {
                                        let i = next_chain_idx;
                                        next_chain_idx += 1;
                                        i
                                    });
                                    chain_color(*idx)
                                }
                                ColorSpec::SecondaryStructure => {
                                    let ss = if obj.structure.ss.is_empty() {
                                        crate::structure::atom::SecondaryStructure::Coil
                                    } else {
                                        obj.structure.ss[*atom_idx]
                                    };
                                    ss_color(ss)
                                }
                                ColorSpec::Spectrum | ColorSpec::BFactor => unreachable!(),
                            };
                            obj.atom_colors[*atom_idx] = color;
                        }
                    }
                }
            }

            // Color changes atom_colors which is baked into all geometry
            (CommandResponse::Ok(format!("Colored {count} atoms")), SceneDirty::ALL)
        }

        Command::Enable(name) => {
            if let Some(obj) = scene.get_mut(&name) {
                obj.visible = true;
                (CommandResponse::Ok(format!("Enabled '{name}'")), SceneDirty::ALL)
            } else {
                (CommandResponse::Error(format!("Object '{name}' not found")), SceneDirty::NONE)
            }
        }

        Command::Disable(name) => {
            if let Some(obj) = scene.get_mut(&name) {
                obj.visible = false;
                (CommandResponse::Ok(format!("Disabled '{name}'")), SceneDirty::ALL)
            } else {
                (CommandResponse::Error(format!("Object '{name}' not found")), SceneDirty::NONE)
            }
        }

        Command::Delete(name) => {
            if scene.remove(&name).is_some() {
                scene.selections.retain(|_, refs| {
                    refs.retain(|(obj, _)| obj != &name);
                    true
                });
                (CommandResponse::Ok(format!("Deleted '{name}'")), SceneDirty::ALL)
            } else {
                (CommandResponse::Error(format!("Object '{name}' not found")), SceneDirty::NONE)
            }
        }

        Command::Zoom { sel } => {
            let atoms = match sel.as_deref() {
                Some(s) => {
                    if scene.selections.contains_key(s) {
                        scene.resolve_selection(s)
                    } else {
                        match crate::command::selection::parse(s) {
                            Ok(expr) => collect_matching(scene, &expr),
                            Err(e) => return (CommandResponse::Error(e), SceneDirty::NONE),
                        }
                    }
                }
                None => scene.all_atoms(),
            };
            if atoms.is_empty() {
                return (CommandResponse::Error("no atoms to zoom to".into()), SceneDirty::NONE);
            }
            // Compute bounding box and update camera
            let mut min = glam::Vec3::splat(f32::MAX);
            let mut max = glam::Vec3::splat(f32::MIN);
            for (obj_name, idx) in &atoms {
                if let Some(obj) = scene.get(obj_name) {
                    let p = obj.structure.atoms[*idx].position;
                    min = min.min(p);
                    max = max.max(p);
                }
            }
            let center  = (min + max) * 0.5;
            let radius  = ((max - min).length() * 0.5).max(1.0);
            camera.center   = center;
            camera.distance = radius * 2.5;
            (CommandResponse::Ok(String::new()), SceneDirty::NONE)
        }

        Command::Reset => {
            // Re-center on all atoms
            let all = scene.all_atoms();
            if !all.is_empty() {
                let mut sum = glam::Vec3::ZERO;
                let mut min = glam::Vec3::splat(f32::MAX);
                let mut max = glam::Vec3::splat(f32::MIN);
                for (obj_name, idx) in &all {
                    if let Some(obj) = scene.get(obj_name) {
                        let p = obj.structure.atoms[*idx].position;
                        sum += p;
                        min = min.min(p);
                        max = max.max(p);
                    }
                }
                camera.center   = sum / all.len() as f32;
                camera.distance = ((max - min).length() * 0.5).max(1.0) * 2.5;
                camera.rotation = glam::Quat::IDENTITY;
            }
            (CommandResponse::Ok(String::new()), SceneDirty::NONE)
        }

        Command::Background(_) => (CommandResponse::Ok(String::new()), SceneDirty::NONE),
        Command::Light { .. }  => (CommandResponse::Ok(String::new()), SceneDirty::NONE),
        Command::Light2 { .. } => (CommandResponse::Ok(String::new()), SceneDirty::NONE),
        Command::Set { .. }    => (CommandResponse::Ok(String::new()), SceneDirty::NONE),
        Command::SetColor { .. } => (CommandResponse::Ok(String::new()), SceneDirty::NONE),
        Command::Get { .. }   => (CommandResponse::Ok(String::new()), SceneDirty::NONE),
        Command::Png { .. } => (CommandResponse::Ok(String::new()), SceneDirty::NONE),
        Command::DockTrace { .. } => (CommandResponse::Ok(String::new()), SceneDirty::NONE),
        Command::DockTraceNav(_) => (CommandResponse::Ok(String::new()), SceneDirty::NONE),
        Command::Help => (CommandResponse::Ok(help_text()), SceneDirty::NONE),
        Command::Quit => (CommandResponse::Ok("bye".into()), SceneDirty::NONE),
    }
}

/// Map a representation type to the appropriate dirty flags.
fn dirty_for_repr(repr: RepresentationType) -> SceneDirty {
    use crate::scene::object::RepresentationType::*;
    match repr {
        BallAndStick | Stick | Backbone | Lines => SceneDirty::ATOMS,
        // Ribbon gaps emit dashed cylinders → also rebuild atom/cylinder buffers.
        // Ghost spheres depend on REP_RIBBON flag.
        Ribbon => SceneDirty::ATOMS | SceneDirty::RIBBON,
        // Ghost spheres depend on REP_SURFACE flag.
        Surface => SceneDirty::ATOMS | SceneDirty::SURFACE,
    }
}

fn help_text() -> String {
    format!("\
─────────────────────────────────────────────────────────────────
 RusMol {version} — command reference
─────────────────────────────────────────────────────────────────
 Commands are entered at the `RusMol>` prompt. Arguments after the
 first are separated by commas. `sel` is an optional selection
 expression; when omitted, a command applies to everything.

 Files
   load <path> [, name]          Load a PDB/PDBQT file",
        version = env!("CARGO_PKG_VERSION"))
    + "

 Display
   show <repr> [, sel]           Show a representation
   hide <repr> [, sel]           Hide a representation
   enable <obj>                  Show an entire object
   disable <obj>                 Hide an entire object
   delete <obj>                  Delete an object

   representations: ribbon / cartoon  ball_stick / bs  stick
                    backbone / trace  surface  lines

 Selection
   select [name,] <expr>         Create a selection set (default name: sele)
   sel    [name,] <expr>         Same as above (shorthand)

   expressions: all  chain <C>  resn <name>  resi <num|range>
                name <atom>  elem <E>  hetatm  not  and  or  ( )
                e.g. chain A and resn ALA
                     resi 1-50 or (hetatm and not resn HOH)

 Color
   color <spec> [, sel]          Change color
   colour <spec> [, sel]         Same as above (alternate spelling)

   color spec: element / cpk         CPK element colors
               chain / chainbows     per-chain colors
               ss / secondary        secondary structure (helix=red, sheet=yellow, coil=grey)
               spectrum / rainbow    N-term(blue) → C-term(red) rainbow
               b / bfactor           B-factor (blue=low → white → red=high)
               red  green  blue  white  black  yellow  orange
               purple  cyan  grey  pink  salmon  teal  marine  forest

 Camera
   zoom [sel]                    Fit selection / whole scene
   reset                         Reset camera to the default position

 Lighting
   light [intensity <f>] [elevation <f>] [azimuth <f>]
                                 Adjust light intensity, elevation, and azimuth

 Settings
   set transparency, <0-1>       Surface transparency (0=opaque)
   set edge_strength, <f>        Edge highlight (0=off, 1=default)
   set roughness, <0-1>          PBR roughness (0=mirror, 1=fully diffuse)
   set metallic, <0-1>           PBR metallic
   set ibl_intensity, <f>        Ambient (IBL) intensity (0=off, 1=default)
   set shadow_strength, <0-1>    Shadow strength (0=none, 1=max)  [shadow]
   set bloom_threshold, <f>      Bloom threshold (default=1.0, lower → wider glow)
   set bloom_intensity, <f>      Bloom intensity (default=0.15)  [bloom]
   set surface_type, <type>     Surface method (gaussian / ses)
   set surface_quality, <0.2-2> Grid resolution in Å (smaller=finer, default=0.5)
   set surface_smooth, <0-100>  Surface smoothing iterations (higher=smoother, default=6)
   set surface_color, <color> [, obj]   Fix the surface color
   set cartoon_color, <color> [, obj]   Fix the ribbon color
   set surface_color, default [, obj]   Restore the original atom colors

 Background
   bg <color>                    Change background color (e.g. bg white / bg black)
   background <color>            Same as above

 Query
   get                           List all parameters
   get <name>                    Show the value of one parameter

 Image export
   png <filename>                Save a screenshot as PNG

 Docking trace
   docktrace <trace>, <ligand>   Load a trace file and enter interactive mode
                                 (prompt: n=next  r=previous  <number>=go to row  q=quit)

 Misc
   help / h / ?                  Show this help
   quit / q / exit               Quit

 Presets (GUI toolbar)
   Default                       Ribbon (SS) + ligand ball-and-stick
   Chain Surface                 Per-chain surface + ligand ball-and-stick
   Binding Site                  Ligand + protein residues within 4 A as sticks,
                                 ligand-protein H-bonds as yellow dashes
   Pocket Surface                Ligand + ligand-facing protein pocket surface

 Mouse
   left drag                     Rotate (arcball)
   right drag                    Pan (translate)
   scroll wheel                  Zoom in / out
   left click                    Pick an atom / residue (prints its identity)

 Keyboard
   Esc                           Quit

 Command line
   rusmol <file> [more files ...]          Open one or more structures
   rusmol <file> -c \"cmd1; cmd2; ...\"       Run commands on startup, keep prompt
   rusmol --help                           Full command-line usage

 Examples
   show surface, chain A         Surface for chain A only
   color ss                      Color by secondary structure
   color spectrum, chain A       Rainbow chain A from N- to C-terminus
   select lig, hetatm and not resn HOH    Name a ligand selection
   color yellow, lig             Color the named selection
   set transparency, 0.4         Make the surface semi-transparent
─────────────────────────────────────────────────────────────────"
}

/// Build a human-readable chain + ligand summary for a freshly loaded structure.
pub fn structure_summary(s: &crate::structure::atom::Structure) -> String {
    use std::collections::{BTreeMap, BTreeSet};

    const AMINO_ACIDS: &[&str] = &[
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        // modified amino acids
        "MSE","SEC","HYP","CME","MLY",
    ];
    const NUCLEICS: &[&str] = &["DA","DC","DG","DT","DI","A","C","G","T","U","I"];
    const WATERS: &[&str] = &["HOH","WAT","DOD"];

    // ── Chains (ATOM records) ────────────────────────────────────────────────
    // chain → (residue_count, mol_type)
    // Use BTreeMap so output is sorted by chain letter.
    let mut chain_residues: BTreeMap<char, BTreeSet<(i32, Option<char>)>> = BTreeMap::new();
    let mut chain_has_protein: BTreeSet<char> = BTreeSet::new();
    let mut chain_has_nucleic: BTreeSet<char> = BTreeSet::new();

    for atom in &s.atoms {
        if atom.is_hetatm { continue; }
        let ch = atom.residue.chain;
        chain_residues
            .entry(ch)
            .or_default()
            .insert((atom.residue.seq_num, atom.residue.ins_code));
        let rn = atom.residue.name.trim();
        if AMINO_ACIDS.contains(&rn) { chain_has_protein.insert(ch); }
        if NUCLEICS.contains(&rn)    { chain_has_nucleic.insert(ch); }
    }

    // ── Ligands (HETATM, non-water) ─────────────────────────────────────────
    // key = (chain, resname, seq_num, ins_code)
    // We collect unique (chain, resname) groups and the residue numbers within each,
    // plus a per-residue heavy-atom (non-hydrogen) count.
    let mut ligand_entries: BTreeMap<(char, String), BTreeSet<(i32, Option<char>)>> =
        BTreeMap::new();
    let mut ligand_heavy: BTreeMap<(char, String, i32, Option<char>), usize> = BTreeMap::new();

    for atom in &s.atoms {
        if !atom.is_hetatm { continue; }
        let rn = atom.residue.name.trim();
        if WATERS.contains(&rn) { continue; }
        ligand_entries
            .entry((atom.residue.chain, rn.to_string()))
            .or_default()
            .insert((atom.residue.seq_num, atom.residue.ins_code));
        if atom.element.trim() != "H" {
            *ligand_heavy
                .entry((atom.residue.chain, rn.to_string(),
                        atom.residue.seq_num, atom.residue.ins_code))
                .or_insert(0) += 1;
        }
    }

    // ── Format ──────────────────────────────────────────────────────────────
    // Build the content lines first (section headers at column 0, entries
    // indented two spaces), then prefix each with a "| " gutter so the block
    // reads as one grouped unit under the per-file header printed by the caller.
    // ASCII only.
    let mut lines: Vec<String> = Vec::new();

    if !chain_residues.is_empty() {
        lines.push("- Chains".to_string());
        for (ch, residues) in &chain_residues {
            let mol_type = if chain_has_protein.contains(ch) {
                "Protein"
            } else if chain_has_nucleic.contains(ch) {
                "Nucleic acid"
            } else {
                "Other"
            };
            let first = residues.iter().next().map(|(n, _)| *n).unwrap_or(0);
            let last  = residues.iter().next_back().map(|(n, _)| *n).unwrap_or(0);
            // Residues with an insertion code (e.g. 1A, 1B) fall inside the
            // numeric range, so note their count so the span reflects the true
            // residue total (the "N residues" column would be redundant here).
            let n_ins = residues.iter().filter(|(_, ins)| ins.is_some()).count();
            let range = match (first == last, n_ins) {
                (true,  0) => format!("{first}"),
                (true,  n) => format!("{first} (+{n} ins)"),
                (false, 0) => format!("{first}-{last}"),
                (false, n) => format!("{first}-{last} (+{n} ins)"),
            };
            // Pad the range column only when a molecule name follows, so lines
            // without a name don't carry trailing whitespace.
            match s.compnd.get(ch) {
                Some(name) => lines.push(format!(
                    "  {ch}   {mol_type:<13} {range:<18}  {name}",
                )),
                None => lines.push(format!(
                    "  {ch}   {mol_type:<13} {range}",
                )),
            }
        }
    }

    if !ligand_entries.is_empty() {
        lines.push("- Ligands".to_string());
        for ((_ch, resname), seqnums) in &ligand_entries {
            let nums: Vec<String> = seqnums
                .iter()
                .map(|(seq, ins)| match ins {
                    Some(ic) => format!("{_ch}:{seq}{ic}"),
                    None     => format!("{_ch}:{seq}"),
                })
                .collect();
            // Heavy-atom count for one instance of this ligand (hydrogens excluded).
            let heavy = seqnums
                .iter()
                .next()
                .and_then(|(seq, ins)| {
                    ligand_heavy.get(&(*_ch, resname.clone(), *seq, *ins)).copied()
                })
                .unwrap_or(0);
            let noun = if heavy == 1 { "atom" } else { "atoms" };
            let count = format!("({heavy} {noun})");
            // Chemical name from HETNAM, synonym from HETSYN
            let chem_name = s.hetnam.get(resname.as_str());
            let synonym   = s.hetsyn.get(resname.as_str());
            let name_str = match (chem_name, synonym) {
                (Some(n), Some(syn)) => format!("  {n} ({syn})"),
                (Some(n), None)      => format!("  {n}"),
                _                    => String::new(),
            };
            // Pad the count column only when a name follows, so lines without a
            // chemical name don't carry trailing whitespace.
            let positions = nums.join(", ");
            if name_str.is_empty() {
                lines.push(format!("  {resname:<5} {positions:<18} {count}"));
            } else {
                lines.push(format!("  {resname:<5} {positions:<18} {count:<11}{name_str}"));
            }
        }
    }

    // Prefix each line with a "|" gutter so the whole block reads as one unit.
    // Section headers start with "- " (→ "|- Chains"); entries are indented two
    // spaces (→ "|   A ...").
    lines
        .iter()
        .map(|line| format!("|{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Collect all (obj_name, atom_idx) pairs matching `expr` across all visible objects.
fn collect_matching(scene: &Scene, expr: &SelectionExpr) -> Vec<(String, usize)> {
    scene
        .iter()
        .flat_map(|(name, obj)| {
            obj.structure
                .atoms
                .iter()
                .enumerate()
                .filter(|(_, atom)| sel_matches(expr, atom, obj))
                .map(|(i, _)| (name.clone(), i))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Return names of objects that contain atoms in the selection (or all objects if sel is None).
/// Rainbow gradient: t=0 → blue, t=1 → red (hue 240°→0°).
fn spectrum_color(t: f32) -> [f32; 3] {
    // HSV → RGB with S=1, V=1, H goes from 240 to 0 degrees
    let h = 240.0 * (1.0 - t.clamp(0.0, 1.0));
    let h6 = h / 60.0;
    let i = h6 as u32;
    let f = h6 - i as f32;
    match i {
        0 => [1.0, f,   0.0],
        1 => [1.0 - f, 1.0, 0.0],
        2 => [0.0, 1.0, f],
        3 => [0.0, 1.0 - f, 1.0],
        4 => [f,   0.0, 1.0],
        _ => [1.0, 0.0, 1.0 - f],
    }
}

/// B-factor color: blue (low) → white (mid) → red (high).
fn bfactor_color(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        let u = t * 2.0;
        [u, u, 1.0]          // blue → white
    } else {
        let u = (t - 0.5) * 2.0;
        [1.0, 1.0 - u, 1.0 - u] // white → red
    }
}

/// Apply spectrum (N→C rainbow) coloring to the given atom list.
/// Groups atoms by (obj, chain), sorts by (seq_num, ins_code), then assigns gradient.
fn apply_spectrum(scene: &mut Scene, atoms: &[(String, usize)]) {
    use std::collections::HashMap;

    // Build (obj_name, chain) → sorted list of (seq_num, ins_code, atom_idx)
    let mut groups: HashMap<(String, char), Vec<(i32, Option<char>, usize)>> = HashMap::new();
    for (obj_name, atom_idx) in atoms {
        if let Some(obj) = scene.get(obj_name) {
            let res = &obj.structure.atoms[*atom_idx].residue;
            groups
                .entry((obj_name.clone(), res.chain))
                .or_default()
                .push((res.seq_num, res.ins_code, *atom_idx));
        }
    }
    for entries in groups.values_mut() {
        entries.sort_unstable_by_key(|&(seq, ins, _)| (seq, ins));
    }

    // Assign colors
    for ((obj_name, _), entries) in &groups {
        let n = entries.len();
        if let Some(obj) = scene.get_mut(obj_name) {
            for (i, &(_, _, atom_idx)) in entries.iter().enumerate() {
                if atom_idx < obj.atom_colors.len() {
                    let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.5 };
                    obj.atom_colors[atom_idx] = spectrum_color(t);
                }
            }
        }
    }
}

/// Apply B-factor coloring to the given atom list.
fn apply_bfactor(scene: &mut Scene, atoms: &[(String, usize)]) {
    // Find min/max temp_factor in selection
    let mut min_b = f32::MAX;
    let mut max_b = f32::MIN;
    for (obj_name, atom_idx) in atoms {
        if let Some(obj) = scene.get(obj_name) {
            let b = obj.structure.atoms[*atom_idx].temp_factor;
            if b < min_b { min_b = b; }
            if b > max_b { max_b = b; }
        }
    }
    let range = (max_b - min_b).max(1e-6);

    for (obj_name, atom_idx) in atoms {
        if let Some(obj) = scene.get_mut(obj_name) {
            if *atom_idx < obj.atom_colors.len() {
                let b = obj.structure.atoms[*atom_idx].temp_factor;
                let t = (b - min_b) / range;
                obj.atom_colors[*atom_idx] = bfactor_color(t);
            }
        }
    }
}

/// Collect all (obj_name, atom_idx) pairs for a selection string.
/// None means all atoms in all objects.
fn atoms_for_sel(scene: &Scene, sel: Option<&str>) -> Vec<(String, usize)> {
    match sel {
        None => scene.all_atoms(),
        Some(s) => {
            if scene.selections.contains_key(s) {
                scene.resolve_selection(s)
            } else {
                match crate::command::selection::parse(s) {
                    Ok(expr) => collect_matching(scene, &expr),
                    Err(_) => vec![],
                }
            }
        }
    }
}
