use anyhow::{Context, Result, bail};
use glam::{Mat3, Vec3};
use std::path::Path;

// ── Trace file types ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TraceHeader {
    pub box_center: Vec3,
    pub box_size: Vec3,
    pub torsion_defs: Vec<TorsionDef>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TorsionDef {
    pub index: usize,
    pub serial_a: u32,
    pub serial_b: u32,
    pub atom_name_a: String,
    pub atom_name_b: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TraceStep {
    pub run: u32,
    pub step: u32,
    pub energy: f32,
    pub pos: Vec3,
    pub rot: Vec3,
    pub torsions: Vec<f32>,
    pub phase: String,
    pub greater_active: String,
    pub accepted: String,
}

// ── Ligand tree types ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LigandAtom {
    pub serial: u32,
    pub name: String,
    pub position: Vec3,
    pub element: String,
    pub is_hetatm: bool,
    pub residue_name: String,
    pub chain: char,
    pub seq_num: i32,
}

#[derive(Debug, Clone)]
pub struct LigandBranch {
    pub anchor_serial: u32,
    pub mobile_serial: u32,
    pub atom_indices: Vec<usize>,
    pub children: Vec<usize>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LigandTree {
    pub atoms: Vec<LigandAtom>,
    pub root_atom_indices: Vec<usize>,
    pub branches: Vec<LigandBranch>,
    pub root_center: Vec3,
    serial_to_index: std::collections::HashMap<u32, usize>,
}

// ── Dock trace state ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DockTrace {
    pub header: TraceHeader,
    pub steps: Vec<TraceStep>,
    pub ligand: LigandTree,
    pub current_step: usize,
}

impl DockTrace {
    pub fn load(trace_path: &Path, ligand_path: &Path) -> Result<Self> {
        let (header, steps) = parse_trace_file(trace_path)?;
        let ligand = parse_ligand_pdbqt(ligand_path)?;

        if steps.is_empty() {
            bail!("trace file contains no steps");
        }

        let n_torsions = header.torsion_defs.len();
        if let Some(s) = steps.first() {
            if s.torsions.len() != n_torsions {
                bail!(
                    "torsion count mismatch: header defines {} but first step has {}",
                    n_torsions,
                    s.torsions.len()
                );
            }
        }

        Ok(Self {
            header,
            steps,
            ligand,
            current_step: 0,
        })
    }

    pub fn total_steps(&self) -> usize {
        self.steps.len()
    }

    pub fn current(&self) -> &TraceStep {
        &self.steps[self.current_step]
    }

    pub fn step_info(&self) -> String {
        let s = self.current();
        format!(
            "row={}/{} run={} step={} energy={:.4} phase={} accepted={}",
            self.current_step + 1, self.steps.len(),
            s.run, s.step, s.energy, s.phase, s.accepted
        )
    }

    pub fn next(&mut self) -> bool {
        if self.current_step + 1 < self.steps.len() {
            self.current_step += 1;
            true
        } else {
            false
        }
    }

    pub fn prev(&mut self) -> bool {
        if self.current_step > 0 {
            self.current_step -= 1;
            true
        } else {
            false
        }
    }

    pub fn reconstruct_positions(&self) -> Vec<Vec3> {
        reconstruct_coordinates(&self.ligand, &self.header, self.current())
    }
}

// ── Trace file parser ───────────────────────────────────────────────────────

fn parse_trace_file(path: &Path) -> Result<(TraceHeader, Vec<TraceStep>)> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read trace file: {}", path.display()))?;

    let mut box_center = Vec3::ZERO;
    let mut box_size = Vec3::ZERO;
    let mut torsion_defs = Vec::new();
    let mut steps = Vec::new();
    let mut header_done = false;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with('#') {
            if let Some(rest) = line.strip_prefix("# BOX center ") {
                let nums: Vec<f32> = rest
                    .split_whitespace()
                    .filter(|s| *s != "size")
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if nums.len() >= 6 {
                    box_center = Vec3::new(nums[0], nums[1], nums[2]);
                    box_size = Vec3::new(nums[3], nums[4], nums[5]);
                }
            } else if let Some(rest) = line.strip_prefix("# TORSION ") {
                if let Some(def) = parse_torsion_def(rest) {
                    torsion_defs.push(def);
                }
            }
            continue;
        }

        if !header_done {
            header_done = true;
            continue;
        }

        if let Some(step) = parse_trace_step(line) {
            steps.push(step);
        }
    }

    torsion_defs.sort_by_key(|t| t.index);

    let header = TraceHeader {
        box_center,
        box_size,
        torsion_defs,
    };

    Ok((header, steps))
}

fn parse_torsion_def(rest: &str) -> Option<TorsionDef> {
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.len() < 5 {
        return None;
    }
    let index: usize = parts[0].parse().ok()?;
    let serials_str = parts[1]; // "serials"
    if serials_str != "serials" {
        return None;
    }
    let serial_pair = parts[2]; // "7-11"
    let (sa, sb) = serial_pair.split_once('-')?;
    let serial_a: u32 = sa.parse().ok()?;
    let serial_b: u32 = sb.parse().ok()?;
    let atoms_str = parts.get(4).unwrap_or(&""); // "N7-C17"
    let (name_a, name_b) = atoms_str.split_once('-').unwrap_or((atoms_str, ""));
    Some(TorsionDef {
        index: index - 1,
        serial_a,
        serial_b,
        atom_name_a: name_a.to_string(),
        atom_name_b: name_b.to_string(),
    })
}

fn parse_trace_step(line: &str) -> Option<TraceStep> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 {
        return None;
    }
    let run: u32 = fields[0].parse().ok()?;
    let step: u32 = fields[1].parse().ok()?;
    let energy: f32 = fields[2].parse().ok()?;
    let pos_x: f32 = fields[3].parse().ok()?;
    let pos_y: f32 = fields[4].parse().ok()?;
    let pos_z: f32 = fields[5].parse().ok()?;
    let rot_x: f32 = fields[6].parse().ok()?;
    let rot_y: f32 = fields[7].parse().ok()?;
    let rot_z: f32 = fields[8].parse().ok()?;
    let torsions: Vec<f32> = fields[9]
        .split(';')
        .filter_map(|s| s.parse().ok())
        .collect();
    let phase = fields[10].to_string();
    let greater_active = fields.get(11).unwrap_or(&"-").to_string();
    let accepted = fields.get(12).unwrap_or(&"-").to_string();

    Some(TraceStep {
        run,
        step,
        energy,
        pos: Vec3::new(pos_x, pos_y, pos_z),
        rot: Vec3::new(rot_x, rot_y, rot_z),
        torsions,
        phase,
        greater_active,
        accepted,
    })
}

// ── Ligand PDBQT parser ────────────────────────────────────────────────────

fn parse_ligand_pdbqt(path: &Path) -> Result<LigandTree> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read ligand PDBQT: {}", path.display()))?;

    let mut atoms: Vec<LigandAtom> = Vec::new();
    let mut serial_to_index: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut root_atom_indices: Vec<usize> = Vec::new();
    let mut branches: Vec<LigandBranch> = Vec::new();

    // Stack for tracking nested branches: (branch_index)
    let mut branch_stack: Vec<usize> = Vec::new();
    let mut in_root = false;

    for line in content.lines() {
        let record = if line.len() >= 6 { &line[..6] } else { line };
        match record.trim() {
            "ROOT" => {
                in_root = true;
            }
            "ENDROO" => {
                in_root = false;
            }
            "BRANCH" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let anchor: u32 = parts[1].parse().unwrap_or(0);
                    let mobile: u32 = parts[2].parse().unwrap_or(0);
                    let idx = branches.len();
                    branches.push(LigandBranch {
                        anchor_serial: anchor,
                        mobile_serial: mobile,
                        atom_indices: Vec::new(),
                        children: Vec::new(),
                    });
                    if let Some(&parent_idx) = branch_stack.last() {
                        branches[parent_idx].children.push(idx);
                    }
                    branch_stack.push(idx);
                }
            }
            "ENDBRA" => {
                branch_stack.pop();
            }
            "ATOM" | "HETATM" => {
                if let Some(atom) = parse_ligand_atom_line(line, record.trim() == "HETATM") {
                    let idx = atoms.len();
                    serial_to_index.insert(atom.serial, idx);
                    if in_root {
                        root_atom_indices.push(idx);
                    } else if let Some(&branch_idx) = branch_stack.last() {
                        branches[branch_idx].atom_indices.push(idx);
                    }
                    atoms.push(atom);
                }
            }
            _ => {}
        }
    }

    let root_center = if root_atom_indices.is_empty() {
        let sum: Vec3 = atoms.iter().map(|a| a.position).sum();
        sum / atoms.len().max(1) as f32
    } else {
        let sum: Vec3 = root_atom_indices.iter().map(|&i| atoms[i].position).sum();
        sum / root_atom_indices.len() as f32
    };

    Ok(LigandTree {
        atoms,
        root_atom_indices,
        branches,
        root_center,
        serial_to_index,
    })
}

fn parse_ligand_atom_line(line: &str, is_hetatm: bool) -> Option<LigandAtom> {
    let col = |start: usize, end: usize| -> &str {
        let end = end.min(line.len());
        if start >= line.len() { "" } else { &line[start..end] }
    };

    let serial: u32 = col(6, 11).trim().parse().ok()?;
    let name = col(12, 16).trim().to_string();
    let residue_name = col(17, 20).trim().to_string();
    let chain = col(21, 22).chars().next().unwrap_or('A');
    let seq_num: i32 = col(22, 26).trim().parse().unwrap_or(0);
    let x: f32 = col(30, 38).trim().parse().ok()?;
    let y: f32 = col(38, 46).trim().parse().ok()?;
    let z: f32 = col(46, 54).trim().parse().ok()?;

    let element = crate::structure::pdb::pdbqt_atom_type_to_element(col(77, 80).trim());

    Some(LigandAtom {
        serial,
        name,
        position: Vec3::new(x, y, z),
        element,
        is_hetatm,
        residue_name,
        chain,
        seq_num,
    })
}

// ── Coordinate reconstruction ───────────────────────────────────────────────

fn reconstruct_coordinates(tree: &LigandTree, header: &TraceHeader, step: &TraceStep) -> Vec<Vec3> {
    let mut positions: Vec<Vec3> = tree.atoms.iter().map(|a| a.position).collect();

    // 1. Center root at origin
    for pos in &mut positions {
        *pos -= tree.root_center;
    }

    // 2. Apply torsions (root to leaves)
    // Map torsion defs to branch indices
    let torsion_to_branch: std::collections::HashMap<(u32, u32), usize> = tree
        .branches
        .iter()
        .enumerate()
        .map(|(i, b)| ((b.anchor_serial, b.mobile_serial), i))
        .collect();

    for (ti, tdef) in header.torsion_defs.iter().enumerate() {
        let angle = step.torsions.get(ti).copied().unwrap_or(0.0);
        if let Some(&branch_idx) = torsion_to_branch.get(&(tdef.serial_a, tdef.serial_b)) {
            let affected = collect_branch_atoms(tree, branch_idx);
            let anchor_idx = tree.serial_to_index.get(&tdef.serial_a).copied();
            let mobile_idx = tree.serial_to_index.get(&tdef.serial_b).copied();
            if let (Some(ai), Some(_mi)) = (anchor_idx, mobile_idx) {
                let pivot = positions[ai];
                let axis_end = positions[tree.serial_to_index[&tdef.serial_b]];
                let axis = (axis_end - pivot).normalize_or_zero();
                if axis.length_squared() > 0.01 {
                    let rot = axis_angle_rotation(axis, angle);
                    for &atom_idx in &affected {
                        let p = positions[atom_idx] - pivot;
                        positions[atom_idx] = rot * p + pivot;
                    }
                }
            }
        }
    }

    // 3. Apply rotation (axis-angle)
    let rot_vec = step.rot;
    let rot_angle = rot_vec.length();
    if rot_angle > 1e-6 {
        let rot_axis = rot_vec / rot_angle;
        let rot_mat = axis_angle_rotation(rot_axis, rot_angle);
        for pos in &mut positions {
            *pos = rot_mat * *pos;
        }
    }

    // 4. Translate to position
    for pos in &mut positions {
        *pos += step.pos;
    }

    positions
}

fn collect_branch_atoms(tree: &LigandTree, branch_idx: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut stack = vec![branch_idx];
    while let Some(bi) = stack.pop() {
        let branch = &tree.branches[bi];
        result.extend_from_slice(&branch.atom_indices);
        for &child_idx in &branch.children {
            stack.push(child_idx);
        }
    }
    result
}

fn axis_angle_rotation(axis: Vec3, angle: f32) -> Mat3 {
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    let Vec3 { x, y, z } = axis;
    Mat3::from_cols(
        Vec3::new(t * x * x + c, t * x * y + s * z, t * x * z - s * y),
        Vec3::new(t * x * y - s * z, t * y * y + c, t * y * z + s * x),
        Vec3::new(t * x * z + s * y, t * y * z - s * x, t * z * z + c),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn trace_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/dock_trace/1lpz.tpe_fast.trace")
    }

    fn ligand_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/dock_trace/ligand.pdbqt")
    }

    #[test]
    fn parse_trace_header() {
        let (header, _) = parse_trace_file(&trace_path()).unwrap();
        assert!(header.box_size.x > 0.0, "box size should be positive");
        assert_eq!(header.torsion_defs.len(), 7, "expected 7 torsions");
    }

    #[test]
    fn parse_trace_steps() {
        let (_, steps) = parse_trace_file(&trace_path()).unwrap();
        assert!(steps.len() > 5000, "expected >5000 steps, got {}", steps.len());
        assert_eq!(steps[0].torsions.len(), 7);
    }

    #[test]
    fn parse_ligand_tree() {
        let tree = parse_ligand_pdbqt(&ligand_path()).unwrap();
        assert_eq!(tree.atoms.len(), 37, "ligand should have 37 atoms");
        assert_eq!(tree.branches.len(), 7, "ligand should have 7 branches");
        assert!(!tree.root_atom_indices.is_empty(), "root should have atoms");
    }

    #[test]
    fn docktrace_load() {
        let dt = DockTrace::load(&trace_path(), &ligand_path()).unwrap();
        assert_eq!(dt.current_step, 0);
        assert!(dt.total_steps() > 0);
    }

    #[test]
    fn reconstruct_returns_correct_count() {
        let dt = DockTrace::load(&trace_path(), &ligand_path()).unwrap();
        let positions = dt.reconstruct_positions();
        assert_eq!(positions.len(), 37);
    }

    #[test]
    fn reconstruct_positions_near_box() {
        let dt = DockTrace::load(&trace_path(), &ligand_path()).unwrap();
        let positions = dt.reconstruct_positions();
        let center = dt.header.box_center;
        let half = dt.header.box_size * 0.5;
        let max_dist = half.length() * 2.0;
        for (i, pos) in positions.iter().enumerate() {
            let d = (*pos - center).length();
            assert!(
                d < max_dist,
                "atom {} at {:.1} Å from box center (max {:.1})",
                i, d, max_dist
            );
        }
    }

    #[test]
    fn navigate_next_prev() {
        let mut dt = DockTrace::load(&trace_path(), &ligand_path()).unwrap();
        assert!(dt.next());
        assert_eq!(dt.current_step, 1);
        assert!(dt.prev());
        assert_eq!(dt.current_step, 0);
        assert!(!dt.prev());
    }
}
