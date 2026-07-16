use glam::Vec3;
use std::collections::HashMap;

use super::atom::{Atom, SecondaryStructure};

type ResKey = (char, i32, Option<char>);

struct Residue {
    key: ResKey,
    n: Vec3,
    #[allow(dead_code)]
    ca: Vec3,
    c: Vec3,
    o: Vec3,
    h: Option<Vec3>,
}

const HBOND_CUTOFF: f32 = -0.5;
const Q1Q2_OVER_F: f32 = 27.888;

/// Assign secondary structure using a DSSP-like hydrogen bond analysis.
/// Returns per-atom SS with the same length as `atoms`.
pub fn assign_ss_dssp(atoms: &[Atom]) -> Vec<SecondaryStructure> {
    let chains = build_chains(atoms);
    let mut residue_ss: HashMap<ResKey, SecondaryStructure> = HashMap::new();

    for chain in &chains {
        if chain.len() < 3 {
            continue;
        }
        let ss = dssp_chain(chain);
        for (res, assignment) in chain.iter().zip(ss) {
            residue_ss.insert(res.key, assignment);
        }
    }

    atoms
        .iter()
        .map(|a| {
            let key = (a.residue.chain, a.residue.seq_num, a.residue.ins_code);
            residue_ss.get(&key).copied().unwrap_or(SecondaryStructure::Coil)
        })
        .collect()
}

fn dssp_chain(chain: &[Residue]) -> Vec<SecondaryStructure> {
    let n = chain.len();
    let mut donor_to = vec![[0i32; 2]; n]; // donor_to[j] = CO targets that NH(j) donates to

    // Initialize
    for i in 0..n {
        donor_to[i] = [-1, -1];
    }

    // Build H-bond map: for each donor j, find the two best acceptor C=O groups
    // DSSP convention: residue j donates NH(j) to C=O of residue i
    let mut best_e: Vec<[f32; 2]> = vec![[0.0; 2]; n];
    let mut best_i: Vec<[i32; 2]> = vec![[-1; 2]; n];

    for j in 1..n {
        let h_pos = match chain[j].h {
            Some(h) => h,
            None => continue,
        };
        for i in 0..n {
            if (i as i32 - j as i32).unsigned_abs() < 2 {
                continue;
            }
            let e = hbond_energy(&chain[i], &chain[j], h_pos);
            if e < HBOND_CUTOFF {
                if e < best_e[j][0] {
                    best_e[j][1] = best_e[j][0];
                    best_i[j][1] = best_i[j][0];
                    best_e[j][0] = e;
                    best_i[j][0] = i as i32;
                } else if e < best_e[j][1] {
                    best_e[j][1] = e;
                    best_i[j][1] = i as i32;
                }
            }
        }
    }

    // Record H-bonds
    for j in 0..n {
        for slot in 0..2 {
            let i = best_i[j][slot];
            if i >= 0 {
                donor_to[j][slot] = i;
            }
        }
    }

    // Helper: does NH(j) donate to CO(i)?
    let has_hbond = |donor_j: usize, acceptor_i: usize| -> bool {
        let target = acceptor_i as i32;
        donor_to[donor_j][0] == target || donor_to[donor_j][1] == target
    };

    // ── Identify n-turns (n = 3, 4, 5) ─────────────────────────────────────
    // Turn(n) at residue i: NH(i+n) → CO(i)
    let mut turn3 = vec![false; n];
    let mut turn4 = vec![false; n];
    let mut turn5 = vec![false; n];

    for i in 0..n {
        if i + 3 < n && has_hbond(i + 3, i) { turn3[i] = true; }
        if i + 4 < n && has_hbond(i + 4, i) { turn4[i] = true; }
        if i + 5 < n && has_hbond(i + 5, i) { turn5[i] = true; }
    }

    // ── Identify β-bridges ──────────────────────────────────────────────────
    #[derive(Clone, Copy, PartialEq)]
    enum BridgeType { None, Parallel, AntiParallel }
    let mut bridge = vec![BridgeType::None; n];
    let mut bridge_partner = vec![0usize; n];

    for i in 1..n.saturating_sub(1) {
        for j in (i + 2)..n.saturating_sub(1) {
            // Parallel bridge: CO(i-1)→NH(j) AND CO(j)→NH(i+1)
            //               OR CO(j-1)→NH(i) AND CO(i)→NH(j+1)
            let parallel =
                (has_hbond(j, i.wrapping_sub(1)) && i > 0 && has_hbond(i + 1, j))
                || (j > 0 && has_hbond(i, j - 1) && j + 1 < n && has_hbond(j + 1, i));

            // Antiparallel bridge: CO(i)→NH(j) AND CO(j)→NH(i)
            //                   OR CO(i-1)→NH(j+1) AND CO(j-1)→NH(i+1)
            let antiparallel =
                (has_hbond(j, i) && has_hbond(i, j))
                || (i > 0 && j + 1 < n && has_hbond(j + 1, i - 1)
                    && j > 0 && i + 1 < n && has_hbond(i + 1, j - 1));

            if parallel || antiparallel {
                let bt = if antiparallel { BridgeType::AntiParallel } else { BridgeType::Parallel };
                if bridge[i] == BridgeType::None {
                    bridge[i] = bt;
                    bridge_partner[i] = j;
                }
                if bridge[j] == BridgeType::None {
                    bridge[j] = bt;
                    bridge_partner[j] = i;
                }
            }
        }
    }

    // ── Assign SS ───────────────────────────────────────────────────────────
    let mut ss = vec![SecondaryStructure::Coil; n];

    // β-strand: residue involved in a bridge
    for i in 0..n {
        if bridge[i] != BridgeType::None {
            ss[i] = SecondaryStructure::Sheet;
        }
    }

    // Extend sheets: isolated bridge residues with a neighbor also bridging
    // form a ladder → label the connecting residues as sheet too
    for i in 1..n {
        if bridge[i] != BridgeType::None && bridge[i - 1] != BridgeType::None {
            ss[i] = SecondaryStructure::Sheet;
            ss[i - 1] = SecondaryStructure::Sheet;
        }
    }

    // α-helix: two consecutive 4-turns → residues i+1..i+4 are helix
    for i in 0..n.saturating_sub(1) {
        if turn4[i] && i + 1 < n && turn4.get(i + 1).copied().unwrap_or(false) {
            for k in (i + 1)..=(i + 4).min(n - 1) {
                if ss[k] != SecondaryStructure::Sheet {
                    ss[k] = SecondaryStructure::Helix;
                }
            }
        }
    }
    // Extend: single 4-turn adjacent to helix
    for i in 0..n {
        if turn4[i] {
            let any_helix = (i + 1..=(i + 4).min(n - 1)).any(|k| ss[k] == SecondaryStructure::Helix);
            if any_helix {
                for k in (i + 1)..=(i + 4).min(n - 1) {
                    if ss[k] != SecondaryStructure::Sheet {
                        ss[k] = SecondaryStructure::Helix;
                    }
                }
            }
        }
    }

    // 3₁₀-helix: two consecutive 3-turns, not already helix or sheet
    for i in 0..n.saturating_sub(1) {
        if turn3[i] && turn3.get(i + 1).copied().unwrap_or(false) {
            for k in (i + 1)..=(i + 3).min(n - 1) {
                if ss[k] == SecondaryStructure::Coil {
                    ss[k] = SecondaryStructure::Helix310;
                }
            }
        }
    }

    // Remove very short helix runs (< 4 residues for α, < 3 for 3₁₀)
    remove_short_runs(&mut ss, SecondaryStructure::Helix, 4);
    remove_short_runs(&mut ss, SecondaryStructure::Helix310, 3);
    remove_short_runs(&mut ss, SecondaryStructure::Sheet, 2);

    ss
}

/// DSSP hydrogen bond energy: E = Q1*Q2/F * (1/rON + 1/rCH - 1/rOH - 1/rCN)
/// Returns energy in kcal/mol. Negative = favorable.
fn hbond_energy(acceptor: &Residue, _donor: &Residue, h_pos: Vec3) -> f32 {
    let r_on = acceptor.o.distance(_donor.n).max(0.5);
    let r_ch = acceptor.c.distance(h_pos).max(0.5);
    let r_oh = acceptor.o.distance(h_pos).max(0.5);
    let r_cn = acceptor.c.distance(_donor.n).max(0.5);

    Q1Q2_OVER_F * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)
}

/// Build per-chain residue lists with backbone atom positions.
/// Estimates H position from geometry when not explicitly present.
fn build_chains(atoms: &[Atom]) -> Vec<Vec<Residue>> {
    // Group backbone atoms by residue
    let mut res_atoms: Vec<(ResKey, HashMap<&str, Vec3>)> = Vec::new();
    let mut seen: HashMap<ResKey, usize> = HashMap::new();

    for atom in atoms {
        if atom.is_hetatm {
            continue;
        }
        let name = atom.name_trimmed();
        if !matches!(name, "N" | "CA" | "C" | "O") {
            continue;
        }
        let key: ResKey = (atom.residue.chain, atom.residue.seq_num, atom.residue.ins_code);
        let idx = if let Some(&i) = seen.get(&key) {
            i
        } else {
            let i = res_atoms.len();
            seen.insert(key, i);
            res_atoms.push((key, HashMap::new()));
            i
        };
        res_atoms[idx].1.insert(name, atom.position);
    }

    // Split into chains (consecutive same-chain residues)
    let mut chains: Vec<Vec<Residue>> = Vec::new();
    let mut current_chain: Option<char> = None;

    for (key, bb) in &res_atoms {
        let (Some(&n), Some(&ca), Some(&c), Some(&o)) =
            (bb.get("N"), bb.get("CA"), bb.get("C"), bb.get("O"))
        else {
            if current_chain.is_some() {
                chains.push(Vec::new());
                current_chain = None;
            }
            continue;
        };

        if current_chain != Some(key.0) {
            chains.push(Vec::new());
            current_chain = Some(key.0);
        }

        let chain_vec = chains.last_mut().unwrap();

        // Standard DSSP H placement: H at 1.008 Å from N,
        // in the direction opposite to previous C=O
        let h = if let Some(prev) = chain_vec.last() {
            let nh_dir = (prev.c - prev.o).normalize_or_zero();
            if nh_dir.length_squared() > 0.01 {
                Some(n + nh_dir * 1.008)
            } else {
                None
            }
        } else {
            None
        };

        chain_vec.push(Residue { key: *key, n, ca, c, o, h });
    }

    chains.into_iter().filter(|c| !c.is_empty()).collect()
}

fn remove_short_runs(assignments: &mut [SecondaryStructure], target: SecondaryStructure, min_len: usize) {
    let n = assignments.len();
    let mut i = 0;
    while i < n {
        if assignments[i] == target {
            let start = i;
            while i < n && assignments[i] == target {
                i += 1;
            }
            if i - start < min_len {
                for j in start..i {
                    assignments[j] = SecondaryStructure::Coil;
                }
            }
        } else {
            i += 1;
        }
    }
}
