use glam::Vec3;
use std::collections::{HashMap, HashSet};

use super::atom::Structure;

#[derive(Debug, Clone)]
pub struct AromaticRing {
    pub atom_indices: Vec<usize>,
    pub center: Vec3,
    pub normal: Vec3,
}

pub fn detect_aromatic_rings(structure: &Structure) -> Vec<AromaticRing> {
    let adj = build_adjacency(&structure.bonds, structure.atoms.len());
    let raw_rings = find_small_rings(&adj, structure.atoms.len());

    let mut result = Vec::new();
    for ring in &raw_rings {
        if let Some(ar) = classify_aromatic(ring, structure) {
            result.push(ar);
        }
    }
    result
}

fn build_adjacency(bonds: &[super::atom::Bond], n: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for b in bonds {
        if b.atom1 < n && b.atom2 < n {
            adj[b.atom1].push(b.atom2);
            adj[b.atom2].push(b.atom1);
        }
    }
    adj
}

fn find_small_rings(adj: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
    let mut rings: Vec<Vec<usize>> = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();

    for start in 0..n {
        find_rings_from(start, adj, &mut rings, &mut seen);
    }
    rings
}

fn find_rings_from(
    start: usize,
    adj: &[Vec<usize>],
    rings: &mut Vec<Vec<usize>>,
    seen: &mut HashSet<Vec<usize>>,
) {
    // DFS with path tracking, max depth 6
    let mut stack: Vec<(usize, usize, Vec<usize>)> = Vec::new(); // (current, parent, path)
    stack.push((start, usize::MAX, vec![start]));

    while let Some((cur, parent, path)) = stack.pop() {
        if path.len() > 6 {
            continue;
        }
        for &next in &adj[cur] {
            if next == parent {
                continue;
            }
            if next == start && (path.len() == 5 || path.len() == 6) {
                let mut key = path.clone();
                let min_pos = key.iter().enumerate().min_by_key(|&(_, &v)| v).unwrap().0;
                key.rotate_left(min_pos);
                if key.last() < key.get(1) {
                    key[1..].reverse();
                }
                if seen.insert(key.clone()) {
                    rings.push(path.clone());
                }
                continue;
            }
            if path.contains(&next) {
                continue;
            }
            if next < start {
                continue;
            }
            let mut new_path = path.clone();
            new_path.push(next);
            stack.push((next, cur, new_path));
        }
    }
}

fn classify_aromatic(ring: &[usize], structure: &Structure) -> Option<AromaticRing> {
    let atoms = &structure.atoms;

    // All ring atoms must be C, N, O, or S
    for &idx in ring {
        let elem = atoms[idx].element.trim();
        if !matches!(elem, "C" | "N" | "O" | "S") {
            return None;
        }
    }

    // Compute center
    let center: Vec3 = ring.iter().map(|&i| atoms[i].position).sum::<Vec3>() / ring.len() as f32;

    // Check planarity: compute best-fit normal, then check max deviation
    let normal = ring_normal(ring, atoms, center);
    let max_dev = ring
        .iter()
        .map(|&i| (atoms[i].position - center).dot(normal).abs())
        .fold(0.0f32, f32::max);

    if max_dev > 0.3 {
        return None;
    }

    // Check connectivity: each ring atom should have >= 2 bonds within the ring
    // (already guaranteed by ring detection) and total bonds suggesting sp2
    let adj: HashMap<usize, Vec<usize>> = {
        let ring_set: HashSet<usize> = ring.iter().copied().collect();
        let mut m: HashMap<usize, Vec<usize>> = HashMap::new();
        for b in &structure.bonds {
            if ring_set.contains(&b.atom1) {
                m.entry(b.atom1).or_default().push(b.atom2);
            }
            if ring_set.contains(&b.atom2) {
                m.entry(b.atom2).or_default().push(b.atom1);
            }
        }
        m
    };

    // sp2 check: ring atoms should have 2 or 3 total bonds (for C)
    // Allow up to 4 for N (pyridine N may have H + lone pair)
    for &idx in ring {
        let total_bonds = adj.get(&idx).map_or(0, |v| v.len());
        let elem = atoms[idx].element.trim();
        match elem {
            "C" if total_bonds > 3 => return None,
            "N" if total_bonds > 3 => return None,
            _ => {}
        }
    }

    Some(AromaticRing {
        atom_indices: ring.to_vec(),
        center,
        normal: normal.normalize(),
    })
}

fn ring_normal(ring: &[usize], atoms: &[super::atom::Atom], center: Vec3) -> Vec3 {
    // Newell's method for polygon normal
    let n = ring.len();
    let mut normal = Vec3::ZERO;
    for i in 0..n {
        let p0 = atoms[ring[i]].position - center;
        let p1 = atoms[ring[(i + 1) % n]].position - center;
        normal += p0.cross(p1);
    }
    if normal.length_squared() < 1e-12 {
        Vec3::Y
    } else {
        normal.normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::pdb::parse_pdbqt;
    use std::path::PathBuf;

    #[test]
    fn detect_cmb_aromatic_rings() {
        let s = parse_pdbqt(&PathBuf::from("tests/dock_trace/ligand.pdbqt")).unwrap();
        let rings = detect_aromatic_rings(&s);
        // Combretastatin A-4 has 2 aromatic rings (two phenyl rings)
        assert!(
            rings.len() >= 2,
            "expected >= 2 aromatic rings, got {}",
            rings.len()
        );
        for r in &rings {
            assert!(r.atom_indices.len() == 5 || r.atom_indices.len() == 6);
            assert!(r.normal.length() > 0.9);
        }
    }
}
