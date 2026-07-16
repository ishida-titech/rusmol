use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::collections::{BTreeSet, HashMap};
use std::ops::Bound::Excluded;

use crate::scene::object::REP_RIBBON;
use crate::structure::atom::{SecondaryStructure, Structure};

/// Number of Catmull-Rom sub-divisions per Cα–Cα segment.
const N_SUB: usize = 16;
/// Number of vertices in each cross-section ring.
const N_PROF: usize = 24;
/// Number of spline steps that form the β-sheet arrow (≈ last 1 Cα interval).
const N_ARROW_STEPS: usize = N_SUB;

/// Returns true if two consecutive residues (sorted by seq_num) are sequential.
pub fn residues_consecutive(seq1: i32, seq2: i32) -> bool {
    seq2 - seq1 == 1
}

/// A gap between two non-consecutive residues, for dashed-line rendering.
pub struct RibbonGap {
    pub p1: [f32; 3],
    pub p2: [f32; 3],
    pub color1: [f32; 3],
    pub color2: [f32; 3],
}

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
    gaps: &mut Vec<RibbonGap>,
) {
    // Single pass over ALL atoms to collect per-chain residue data.
    // Do NOT use chain_ranges: in PDB files HETATM records (ligands, waters)
    // that share a chain ID with protein atoms appear last and would overwrite
    // the chain_ranges entry for the protein chain.
    //
    // per_chain: chain → { (seq_num, ins_code) → (ca_global_idx, o_global_idx) }
    let mut per_chain: HashMap<char, HashMap<(i32, Option<char>), (usize, Option<usize>)>> =
        HashMap::new();

    // Every backbone (Cα) residue actually present in the input, per chain,
    // regardless of the current representation flags. Used to tell a genuine
    // data gap (missing/disordered residues) from a gap created merely by
    // hiding the ribbon for some residues — only the former gets a dashed line.
    let mut present: HashMap<char, BTreeSet<(i32, Option<char>)>> = HashMap::new();

    for (global_i, atom) in structure.atoms.iter().enumerate() {
        if atom.is_hetatm {
            continue;
        }
        if matches!(atom.alt_loc, Some(c) if c != 'A') {
            continue;
        }
        let chain = atom.residue.chain;
        let key   = (atom.residue.seq_num, atom.residue.ins_code);

        if atom.name.trim() == "CA" {
            present.entry(chain).or_default().insert(key);
        }

        // Ribbon geometry is built only from residues whose ribbon rep is on.
        if atom_rep_show.get(global_i).copied().unwrap_or(0) & REP_RIBBON == 0 {
            continue;
        }
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

        // Split at chain breaks (non-consecutive residue numbers)
        let mut segments: Vec<Vec<(i32, Option<char>, usize, Option<usize>)>> = Vec::new();
        let mut current = vec![sorted[0]];

        let chain_present = present.get(&chain_id);
        for i in 1..sorted.len() {
            if !residues_consecutive(sorted[i - 1].0, sorted[i].0) {
                // The two ribbon-visible residues are non-consecutive. Emit a
                // dashed line only if NO input residue sits between them — i.e.
                // it is a real data gap (disorder / missing residues), not a gap
                // produced by hiding the ribbon for residues that do exist.
                let prev_key = (sorted[i - 1].0, sorted[i - 1].1);
                let curr_key = (sorted[i].0, sorted[i].1);
                let hidden_between = chain_present.is_some_and(|set| {
                    set.range((Excluded(prev_key), Excluded(curr_key))).next().is_some()
                });
                if !hidden_between {
                    let ca_prev = sorted[i - 1].2;
                    let ca_curr = sorted[i].2;
                    gaps.push(RibbonGap {
                        p1: structure.atoms[ca_prev].position.to_array(),
                        p2: structure.atoms[ca_curr].position.to_array(),
                        color1: atom_colors[ca_prev],
                        color2: atom_colors[ca_curr],
                    });
                }
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

    // ── Smooth Cα positions per secondary structure ───────────────────────────
    // Sheet: moderate smoothing removes strand-level undulation.
    // Helix: very light smoothing to reduce per-residue jitter without
    //        collapsing the helical spiral.
    // Coil:  no smoothing — coils follow the actual backbone.
    let mut ca_spline = ca_pos.clone();
    for pass in 0..3 {
        let prev = ca_spline.clone();
        for i in 1..n - 1 {
            let w = match ss_types[i] {
                SecondaryStructure::Sheet    => 0.25,
                SecondaryStructure::Helix    => { if pass < 2 { 0.15 } else { 0.0 } }
                SecondaryStructure::Helix310 => 0.0, // coil-like: no smoothing
                SecondaryStructure::Coil     => 0.0,
            };
            if w > 0.0 {
                let avg = (prev[i - 1] + prev[i] * 2.0 + prev[i + 1]) * 0.25;
                ca_spline[i] = prev[i] * (1.0 - w) + avg * w;
            }
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

    let m = spos.len();

    // ── Pass 2: compute local helix axis at each spline point ────────────
    // For helix regions, the cross-section wide axis should point radially
    // outward from the helix axis.  We compute a local axis per spline point
    // using a sliding window of Cα positions, then derive the radial direction.
    let helix_radial = compute_helix_radial(&spos, &sss, &stan, m);

    // ── Helix boundary taper factor ──────────────────────────────────────
    // Compute a [0,1] factor for each spline point: 0 at helix boundaries,
    // 1 at interior.  Used to taper radial scaling and frame blending.
    let helix_taper = {
        let taper_len = N_SUB * 3; // fade over ~3 Cα intervals
        let mut taper = vec![0.0f32; m];
        let mut i = 0;
        while i < m {
            if sss[i] != SecondaryStructure::Helix { i += 1; continue; }
            let run_start = i;
            while i < m && sss[i] == SecondaryStructure::Helix { i += 1; }
            let run_end = i;
            let run_len = run_end - run_start;
            for j in run_start..run_end {
                let dist_start = j - run_start;
                let dist_end = run_end - 1 - j;
                let dist = dist_start.min(dist_end);
                let t = if taper_len > 0 {
                    (dist as f32 / taper_len as f32).min(1.0)
                } else { 1.0 };
                // Smoothstep for gentle transition
                taper[j] = t * t * (3.0 - 2.0 * t);
                // Also zero out taper if PCA window is too small (< 1 turn)
                let min_window = N_SUB * 3; // ~3 residues
                if dist_start < min_window / 2 && run_start == 0 { /* near segment start */ }
                if run_len < min_window { taper[j] *= run_len as f32 / min_window as f32; }
            }
        }
        taper
    };

    // ── Scale helix spiral radius outward (tapered at boundaries) ────────
    {
        let scale = 0.5; // max additional radial displacement in Å
        for i in 0..m {
            if let Some(r) = helix_radial[i] {
                if sss[i] == SecondaryStructure::Helix {
                    spos[i] += r * scale * helix_taper[i];
                }
            }
        }
    }

    // ── Pass 3: side vector via Parallel Transport + radial correction ───
    //
    // Parallel transport gives a rotation-minimising frame (smooth, no jumps).
    // For helix interiors we use the radial binormal; at boundaries we blend
    // between the parallel-transport frame and the helix frame using the taper.
    let mut sside: Vec<Vec3> = Vec::with_capacity(m);
    {
        let mut side = project_perp(o_dir[0], stan[0]).normalize_or_zero();
        if side.length_squared() < 0.5 { side = orthogonal_to(stan[0]); }

        for i in 0..m {
            let seg = (i / N_SUB).min(n - 1);
            let ss = sss[i];

            // At Cα boundaries: snap for coil / SS-transition points.
            if i > 0 && i % N_SUB == 0 {
                let is_interior_sheet = ss_types[seg] == SecondaryStructure::Sheet
                    && seg > 0
                    && ss_types[seg - 1] == SecondaryStructure::Sheet;
                let is_interior_helix = ss_types[seg] == SecondaryStructure::Helix
                    && seg > 0
                    && ss_types[seg - 1] == SecondaryStructure::Helix;
                if !is_interior_sheet && !is_interior_helix {
                    let o_ideal = project_perp(o_dir[seg], stan[i]).normalize_or_zero();
                    if o_ideal.length_squared() > 0.5 {
                        side = if o_ideal.dot(side) >= 0.0 { o_ideal } else { -o_ideal };
                    }
                }
            }

            // For helix points: blend between parallel-transport frame and
            // helix radial binormal, using taper factor (0 at boundaries, 1 interior).
            if ss == SecondaryStructure::Helix {
                if let Some(radial) = helix_radial[i] {
                    let binormal = stan[i].cross(radial).normalize_or_zero();
                    if binormal.length_squared() > 0.5 {
                        let target = if binormal.dot(side) >= 0.0 { binormal } else { -binormal };
                        let t = helix_taper[i];
                        side = (side * (1.0 - t) + target * t).normalize();
                    }
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

    // ── Smooth cross-section profile dims at SS transitions ──────────────────
    // Within each SS region the width must be exactly constant.  Only at
    // SS boundaries do we interpolate, using a cubic smoothstep over
    // ±TRANSITION_HALF spline points centred on the boundary.
    let mut prof_a: Vec<f32> = sss.iter().map(|&ss| profile_dims(ss).0).collect();
    let mut prof_b: Vec<f32> = sss.iter().map(|&ss| profile_dims(ss).1).collect();
    {
        let tr_half = N_SUB; // half-width of blend zone (≈1 Cα interval)

        // Collect boundary positions where SS type changes
        let bnd: Vec<usize> = (1..m).filter(|&i| sss[i] != sss[i - 1]).collect();

        for (bi, &b) in bnd.iter().enumerate() {
            // If the left side ends with an arrow zone, use the arrow tip dims
            // instead of the raw Sheet dims — otherwise the coil after the arrow
            // tip suddenly bloats from 1.6 back to ~1.0 before settling to 0.22.
            let (a_l, b_l) = if b > 0 && arrow_frac[b - 1] >= 0.0 {
                arrow_profile_dims(prof_a[b - 1], prof_b[b - 1], arrow_frac[b - 1])
            } else {
                profile_dims(sss[b - 1])
            };
            let (a_r, b_r) = profile_dims(sss[b]);

            // Clamp zone to midpoints between adjacent boundaries to prevent overlap
            let left_limit  = if bi > 0            { (bnd[bi - 1] + b) / 2 } else { 0 };
            let right_limit = if bi + 1 < bnd.len() { (b + bnd[bi + 1]) / 2 } else { m };

            let start = b.saturating_sub(tr_half).max(left_limit);
            let end   = (b + tr_half).min(m).min(right_limit);
            if end <= start { continue; }

            let span = (end - start).max(1) as f32;
            for j in start..end {
                if arrow_frac[j] >= 0.0 { continue; } // keep arrow zone intact
                let t = (j - start) as f32 / span;
                let s = t * t * (3.0 - 2.0 * t); // smoothstep
                prof_a[j] = a_l + (a_r - a_l) * s;
                prof_b[j] = b_l + (b_r - b_l) * s;
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
        let (a, b) = if arrow_frac[i] >= 0.0 {
            arrow_profile_dims(prof_a[i], prof_b[i], arrow_frac[i])
        } else {
            (prof_a[i], prof_b[i])
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
        SecondaryStructure::Helix    => (1.6, 0.30),  // wide ribbon, thin profile
        SecondaryStructure::Helix310 => (0.22, 0.22), // coil-like tube (colored as helix)
        SecondaryStructure::Sheet    => (1.6, 0.10),  // very flat ribbon → arrow contrast
        SecondaryStructure::Coil     => (0.22, 0.22),
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

// ── Helix radial direction ───────────────────────────────────────────────────

/// Window radius (in spline points) for local helix axis estimation.
const HELIX_AXIS_WINDOW: usize = 48; // ≈ 3 residues × 16 sub-steps

/// For each spline point in a helix region, compute the radial direction
/// (outward from the local helix axis).  Returns `None` for non-helix points.
///
/// Algorithm: for each helix spline point, take a window of nearby helix
/// spline positions, fit a local axis via PCA (first principal component),
/// then the radial direction = (point - axis_projection).normalize().
fn compute_helix_radial(
    spos: &[Vec3],
    sss: &[SecondaryStructure],
    _stan: &[Vec3],
    m: usize,
) -> Vec<Option<Vec3>> {
    let mut radial: Vec<Option<Vec3>> = vec![None; m];

    // Find contiguous helix runs in spline-point space.
    let mut i = 0;
    while i < m {
        if sss[i] != SecondaryStructure::Helix {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < m && sss[i] == SecondaryStructure::Helix { i += 1; }
        let run_end = i;
        let run_len = run_end - run_start;
        if run_len < 3 { continue; }

        // For each point in the run, compute local axis from a window.
        // Skip points near boundaries where PCA is unreliable (asymmetric window).
        let skip = N_SUB; // ~1 Cα interval (shorter than taper_len so blending starts gently)
        for j in run_start..run_end {
            let dist_start = j - run_start;
            let dist_end = run_end - 1 - j;
            if dist_start < skip || dist_end < skip { continue; }

            let half = HELIX_AXIS_WINDOW;
            let lo = if j >= run_start + half { j - half } else { run_start };
            let hi = (j + half + 1).min(run_end);
            if hi - lo < 3 { continue; }

            let axis = local_pca_axis(&spos[lo..hi]);
            if axis.length_squared() < 0.5 { continue; }

            // Centroid of the window
            let n_pts = (hi - lo) as f32;
            let centroid: Vec3 = spos[lo..hi].iter().copied().sum::<Vec3>() / n_pts;

            // Project current point onto axis, radial = point - projection
            let v = spos[j] - centroid;
            let along = v.dot(axis);
            let on_axis = centroid + axis * along;
            let r = spos[j] - on_axis;
            if r.length_squared() > 1e-6 {
                radial[j] = Some(r.normalize());
            }
        }
    }

    radial
}

/// Compute the principal axis (first principal component) of a set of points
/// using power iteration.
fn local_pca_axis(pts: &[Vec3]) -> Vec3 {
    let n = pts.len() as f32;
    let centroid: Vec3 = pts.iter().copied().sum::<Vec3>() / n;

    // Covariance matrix (symmetric 3×3)
    let mut cov = [[0.0f32; 3]; 3];
    for &p in pts {
        let d = p - centroid;
        let da = d.to_array();
        for r in 0..3 {
            for c in r..3 {
                cov[r][c] += da[r] * da[c];
            }
        }
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // Power iteration (10 steps)
    let mut v = Vec3::new(1.0, 1.0, 1.0).normalize();
    for _ in 0..10 {
        let va = v.to_array();
        let new = Vec3::new(
            cov[0][0] * va[0] + cov[0][1] * va[1] + cov[0][2] * va[2],
            cov[1][0] * va[0] + cov[1][1] * va[1] + cov[1][2] * va[2],
            cov[2][0] * va[0] + cov[2][1] * va[1] + cov[2][2] * va[2],
        );
        let len = new.length();
        if len < 1e-10 { break; }
        v = new / len;
    }
    v
}

