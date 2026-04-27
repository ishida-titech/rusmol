use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::collections::HashMap;

use crate::scene::object::REP_RIBBON;
use crate::structure::atom::{SecondaryStructure, Structure};

/// Number of Catmull-Rom sub-divisions per Cα–Cα segment.
const N_SUB: usize = 16;
/// Number of vertices in each cross-section ring.
const N_PROF: usize = 24;
/// Maximum Cα–Cα distance (Å) before treating as a chain break.
const BREAK_DIST: f32 = 5.0;
/// Number of spline steps that form the β-sheet arrow (≈ last 1 Cα interval).
const N_ARROW_STEPS: usize = N_SUB;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RibbonVertex {
    pub position:   [f32; 3], // offset  0, 12 bytes
    pub normal:     [f32; 3], // offset 12, 12 bytes
    pub color:      [f32; 3], // offset 24, 12 bytes
    pub residue_id: u32,      // offset 36,  4 bytes → total 40 bytes
}

impl RibbonVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<RibbonVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 36, shader_location: 3, format: wgpu::VertexFormat::Uint32   },
            ],
        }
    }
}

/// Build ribbon geometry for the entire scene, appending into the provided vecs.
/// `residue_ids[i]` is the residue identifier for atom `i` (index of the first
/// atom in the same residue). All vertices within one residue share the same value.
pub fn build_ribbon(
    structure: &Structure,
    atom_colors: &[[f32; 3]],
    residue_ids: &[u32],
    atom_rep_show: &[u8],
    vertices: &mut Vec<RibbonVertex>,
    indices: &mut Vec<u32>,
) {
    // Single pass over ALL atoms to collect per-chain residue data.
    // Do NOT use chain_ranges: in PDB files HETATM records (ligands, waters)
    // that share a chain ID with protein atoms appear last and would overwrite
    // the chain_ranges entry for the protein chain.
    //
    // per_chain: chain → { (seq_num, ins_code) → (ca_global_idx, o_global_idx) }
    let mut per_chain: HashMap<char, HashMap<(i32, Option<char>), (usize, Option<usize>)>> =
        HashMap::new();

    for (global_i, atom) in structure.atoms.iter().enumerate() {
        if atom_rep_show.get(global_i).copied().unwrap_or(0) & REP_RIBBON == 0 {
            continue;
        }
        if atom.is_hetatm {
            continue;
        }
        if matches!(atom.alt_loc, Some(c) if c != 'A') {
            continue;
        }
        let chain = atom.residue.chain;
        let key   = (atom.residue.seq_num, atom.residue.ins_code);
        let chain_map = per_chain.entry(chain).or_default();
        match atom.name.trim() {
            "CA" => { chain_map.entry(key).or_insert((global_i, None)).0 = global_i; }
            "O"  => {
                if let Some(entry) = chain_map.get_mut(&key) {
                    entry.1 = Some(global_i);
                }
            }
            _ => {}
        }
    }

    let mut chains: Vec<char> = per_chain.keys().copied().collect();
    chains.sort_unstable();

    for chain_id in chains {
        let residue_map = per_chain.remove(&chain_id).unwrap();

        let mut sorted: Vec<(i32, Option<char>, usize, Option<usize>)> = residue_map
            .into_iter()
            .map(|((seq, ins), (ca, o))| (seq, ins, ca, o))
            .collect();
        sorted.sort_unstable_by_key(|&(seq, ins, _, _)| (seq, ins));

        if sorted.len() < 2 {
            continue;
        }

        // Split at chain breaks (CA–CA distance > BREAK_DIST)
        let mut segments: Vec<Vec<(i32, Option<char>, usize, Option<usize>)>> = Vec::new();
        let mut current = vec![sorted[0]];

        for i in 1..sorted.len() {
            let p_prev = structure.atoms[sorted[i - 1].2].position;
            let p_curr = structure.atoms[sorted[i].2].position;
            if (p_curr - p_prev).length() > BREAK_DIST {
                segments.push(std::mem::take(&mut current));
            }
            current.push(sorted[i]);
        }
        segments.push(current);

        for seg in segments {
            if seg.len() >= 2 {
                build_segment(structure, atom_colors, residue_ids, &seg, vertices, indices);
            }
        }
    }
}

// ── Per-segment (no chain breaks) ────────────────────────────────────────────

fn build_segment(
    structure: &Structure,
    atom_colors: &[[f32; 3]],
    residue_ids: &[u32],
    residues: &[(i32, Option<char>, usize, Option<usize>)],
    vertices: &mut Vec<RibbonVertex>,
    indices: &mut Vec<u32>,
) {
    let n = residues.len();

    let ca_pos: Vec<Vec3> = residues.iter().map(|r| structure.atoms[r.2].position).collect();
    let colors: Vec<[f32; 3]> = residues.iter().map(|r| atom_colors[r.2]).collect();
    let ss_types: Vec<SecondaryStructure> = residues.iter().map(|r| {
        if structure.ss.is_empty() {
            SecondaryStructure::Coil
        } else {
            structure.ss[r.2]
        }
    }).collect();

    // Compute O-direction vectors and make them consistent (no flipping)
    let mut o_dir: Vec<Vec3> = residues.iter().enumerate().map(|(i, r)| {
        if let Some(o_idx) = r.3 {
            let v = structure.atoms[o_idx].position - ca_pos[i];
            if v.length_squared() > 1e-6 { v.normalize() } else { Vec3::Y }
        } else {
            Vec3::Y
        }
    }).collect();

    // Enforce O-direction consistency: flip to avoid sign changes
    for i in 1..n {
        if o_dir[i].dot(o_dir[i - 1]) < 0.0 {
            o_dir[i] = -o_dir[i];
        }
    }

    // Smooth O-direction vectors: 6 passes of 5-point Gaussian.
    // β-sheet O atoms alternate up/down; without smoothing the interpolated
    // side vector oscillates each residue and the ribbon appears wavy.
    for _ in 0..6 {
        let prev = o_dir.clone();
        // 5-point kernel (weights 1:4:6:4:1) for interior
        for i in 2..n.saturating_sub(2) {
            let avg = prev[i - 2]
                + prev[i - 1] * 4.0
                + prev[i]     * 6.0
                + prev[i + 1] * 4.0
                + prev[i + 2];
            let s = avg.normalize_or_zero();
            if s.length_squared() > 0.5 { o_dir[i] = s; }
        }
        // 3-point kernel for positions near the ends
        if n >= 3 {
            let s = (prev[0] + prev[1] * 2.0 + prev[2]).normalize_or_zero();
            if s.length_squared() > 0.5 { o_dir[1] = s; }
            let s = (prev[n - 3] + prev[n - 2] * 2.0 + prev[n - 1]).normalize_or_zero();
            if s.length_squared() > 0.5 { o_dir[n - 2] = s; }
        }
        // Re-enforce consistency after each smoothing pass
        for i in 1..n {
            if o_dir[i].dot(o_dir[i - 1]) < 0.0 { o_dir[i] = -o_dir[i]; }
        }
    }

    // ── Smooth Cα positions (strand-level undulation removal) ────────────────
    // The β-sheet backbone has an inherent gentle wave at the strand level.
    // Apply a 5-point Gaussian average (6 passes) to the Cα positions used
    // as spline control points. The original positions are kept for o_dir.
    let mut ca_spline = ca_pos.clone();
    for _ in 0..6 {
        let prev = ca_spline.clone();
        for i in 2..n.saturating_sub(2) {
            ca_spline[i] = (prev[i - 2]
                + prev[i - 1] * 4.0
                + prev[i]     * 6.0
                + prev[i + 1] * 4.0
                + prev[i + 2]) * (1.0 / 16.0);
        }
        if n >= 3 {
            ca_spline[1]     = (prev[0] + prev[1] * 2.0 + prev[2])     * (1.0 / 4.0);
            ca_spline[n - 2] = (prev[n - 3] + prev[n - 2] * 2.0 + prev[n - 1]) * (1.0 / 4.0);
        }
    }

    // ── Catmull-Rom spline (pass 1: geometry, colour, SS) ────────────────────
    let total_pts = (n - 1) * N_SUB + 1;
    let mut spos   = Vec::with_capacity(total_pts);
    let mut stan   = Vec::with_capacity(total_pts);
    let mut scol   = Vec::with_capacity(total_pts);
    let mut sss    = Vec::with_capacity(total_pts);
    let mut sresid = Vec::with_capacity(total_pts);

    for seg in 0..n - 1 {
        let p0 = if seg > 0     { ca_spline[seg - 1]         } else { ca_spline[0] * 2.0 - ca_spline[1] };
        let p1 = ca_spline[seg];
        let p2 = ca_spline[seg + 1];
        let p3 = if seg + 2 < n { ca_spline[seg + 2]         } else { ca_spline[n - 1] * 2.0 - ca_spline[n - 2] };

        let steps = if seg == n - 2 { N_SUB + 1 } else { N_SUB };
        for k in 0..steps {
            let t = k as f32 / N_SUB as f32;

            let pos = catmull_rom(p0, p1, p2, p3, t);
            let tan = catmull_rom_deriv(p0, p1, p2, p3, t);
            let tan = if tan.length_squared() > 1e-10 { tan.normalize() } else { Vec3::Y };

            let col    = lerp_color(colors[seg], colors[seg + 1], t);
            let ca_idx = if t < 0.5 { residues[seg].2 } else { residues[seg + 1].2 };
            let ss     = if t < 0.5 { ss_types[seg]   } else { ss_types[seg + 1] };

            sresid.push(if ca_idx < residue_ids.len() { residue_ids[ca_idx] } else { 0 });
            spos.push(pos);
            stan.push(tan);
            scol.push(col);
            sss.push(ss);
        }
    }

    // ── Pass 2: side vector via Parallel Transport + per-Cα O correction ─────
    //
    // Pure interpolation of O directions produces waviness because even a small
    // oscillation in the projected side vector is amplified by the flat ribbon.
    // Parallel transport is rotation-minimising: it only rotates the side vector
    // as much as the tangent rotates, yielding a smooth ribbon.
    // At each Cα position (every N_SUB steps) we snap to the smoothed O direction
    // to prevent cumulative drift without reintroducing oscillation.
    let m = spos.len();
    let mut sside: Vec<Vec3> = Vec::with_capacity(m);
    {
        // Initialise from the first O direction
        let mut side = project_perp(o_dir[0], stan[0]).normalize_or_zero();
        if side.length_squared() < 0.5 { side = orthogonal_to(stan[0]); }

        for i in 0..m {
            // At each Cα boundary (except the very first point which is already set),
            // snap the side vector to the smoothed O direction.
            if i > 0 && i % N_SUB == 0 {
                let seg = (i / N_SUB).min(n - 1);
                let o_ideal = project_perp(o_dir[seg], stan[i]).normalize_or_zero();
                if o_ideal.length_squared() > 0.5 {
                    // Flip to maintain sign consistency with the propagated frame.
                    side = if o_ideal.dot(side) >= 0.0 { o_ideal } else { -o_ideal };
                }
            }

            sside.push(side);

            // Parallel-transport side to the next tangent.
            if i + 1 < m {
                let s = project_perp(side, stan[i + 1]).normalize_or_zero();
                if s.length_squared() > 0.5 { side = s; }
            }
        }

        // Fix any degenerate entries by inheriting from the previous point.
        for i in 1..sside.len() {
            if sside[i].length_squared() < 1e-10 { sside[i] = sside[i - 1]; }
        }
    }

    // ── β-sheet arrow zone ────────────────────────────────────────────────────
    // Detect the last N_ARROW_STEPS spline points of each Sheet run.
    // arrow_frac[i] = -1.0  → outside arrow zone (use normal profile)
    // arrow_frac[i] ∈ [0, 1] → inside: 0 = arrowhead base (max width), 1 = tip
    let mut arrow_frac: Vec<f32> = vec![-1.0; m];
    {
        let mut i = 0;
        while i < m {
            if sss[i] == SecondaryStructure::Sheet {
                let run_start = i;
                while i < m && sss[i] == SecondaryStructure::Sheet { i += 1; }
                let run_end = i;
                let run_len = run_end - run_start;
                // Arrow covers the last N_ARROW_STEPS, but at most 60% of the run.
                let n_arrow = N_ARROW_STEPS.min(run_len * 3 / 5 + 1);
                if n_arrow >= 2 {
                    let arrow_start = run_end - n_arrow;
                    let denom = (n_arrow - 1).max(1);
                    for j in arrow_start..run_end {
                        arrow_frac[j] = (j - arrow_start) as f32 / denom as f32;
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    // ── Extrude cross-section profile ─────────────────────────────────────────
    let vbase  = vertices.len() as u32;

    for i in 0..m {
        let pos  = spos[i];
        let side = if sside[i].length_squared() > 1e-10 { sside[i].normalize() } else { orthogonal_to(stan[i]) };
        let bi   = stan[i].cross(side).normalize_or_zero();
        let bi   = if bi.length_squared() < 1e-10 { orthogonal_to(stan[i]) } else { bi.normalize() };
        let col  = scol[i];
        let (base_a, base_b) = profile_dims(sss[i]);
        let (a, b) = if arrow_frac[i] >= 0.0 {
            arrow_profile_dims(base_a, base_b, arrow_frac[i])
        } else {
            (base_a, base_b)
        };

        for j in 0..N_PROF {
            let angle = j as f32 * std::f32::consts::TAU / N_PROF as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            let offset = side * (cos_a * a) + bi * (sin_a * b);
            // Clamp semi-axes for normal to avoid division blow-up near the tip.
            let a_n = a.max(0.08);
            let b_n = b.max(0.08);
            let normal = (side * (cos_a / a_n) + bi * (sin_a / b_n)).normalize_or_zero();
            vertices.push(RibbonVertex {
                position:   (pos + offset).to_array(),
                normal:     normal.to_array(),
                color:      col,
                residue_id: sresid[i],
            });
        }
    }

    // Connect adjacent rings with quads (2 triangles each)
    for i in 0..(m - 1) as u32 {
        let r0 = vbase + i * N_PROF as u32;
        let r1 = vbase + (i + 1) * N_PROF as u32;
        for j in 0..N_PROF as u32 {
            let jn = (j + 1) % N_PROF as u32;
            indices.extend_from_slice(&[r0 + j, r1 + j, r0 + jn, r0 + jn, r1 + j, r1 + jn]);
        }
    }
}

// ── Math helpers ─────────────────────────────────────────────────────────────

fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    (p1 * 2.0
        + (p2 - p0) * t
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
        + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3)
        * 0.5
}

fn catmull_rom_deriv(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    ((p2 - p0)
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (2.0 * t)
        + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * (3.0 * t2))
        * 0.5
}

/// Cross-section semi-axes (width, thickness) in Ångströms.
fn profile_dims(ss: SecondaryStructure) -> (f32, f32) {
    match ss {
        SecondaryStructure::Helix => (0.9, 0.70),  // near-circular tube (a:b ≈ 1.3)
        SecondaryStructure::Sheet => (1.6, 0.10),  // very flat ribbon → arrow contrast
        SecondaryStructure::Coil  => (0.22, 0.22),
    }
}

/// Compute the cross-section semi-axes for the β-sheet arrow zone.
/// `frac` ∈ [0, 1]: 0 = arrowhead base (maximum width), 1 = tip (zero width).
///
/// Shape: pure triangular arrowhead — width = max_width × (1 − frac).
/// This produces the classic PyMOL-style β-arrow when viewed from above.
fn arrow_profile_dims(a: f32, b: f32, frac: f32) -> (f32, f32) {
    let max_a = a + 1.2;     // arrowhead base width (e.g. 1.5 + 1.2 = 2.7 Å)
    let new_a = (max_a * (1.0 - frac)).max(0.02);
    let new_b = (b * (1.0 - frac * 0.5)).max(0.02); // thickness tapers half as fast
    (new_a, new_b)
}

/// Project `v` perpendicular to `axis`; returns Vec3::ZERO if degenerate.
fn project_perp(v: Vec3, axis: Vec3) -> Vec3 {
    v - axis * axis.dot(v)
}

/// Return an arbitrary unit vector orthogonal to `v`.
fn orthogonal_to(v: Vec3) -> Vec3 {
    if v.x.abs() <= v.y.abs() && v.x.abs() <= v.z.abs() {
        Vec3::new(0.0, -v.z, v.y).normalize_or_zero()
    } else if v.y.abs() <= v.z.abs() {
        Vec3::new(-v.z, 0.0, v.x).normalize_or_zero()
    } else {
        Vec3::new(-v.y, v.x, 0.0).normalize_or_zero()
    }
}

fn lerp_color(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}
