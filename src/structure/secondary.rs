use super::atom::{Atom, SecondaryStructure};
use super::dssp::assign_ss_dssp;

pub struct SsRange {
    pub chain: char,
    pub start_seq: i32,
    pub start_ins: Option<char>,
    pub end_seq: i32,
    pub end_ins: Option<char>,
    pub ss: SecondaryStructure,
}

/// Assign secondary structure to each atom.
/// The returned Vec has the same length as `atoms`.
/// When `ranges` is empty (e.g. PDBQT), falls back to DSSP hydrogen-bond analysis.
pub fn assign_ss(atoms: &[Atom], ranges: &[SsRange]) -> Vec<SecondaryStructure> {
    if ranges.is_empty() {
        return assign_ss_dssp(atoms);
    }
    atoms
        .iter()
        .map(|atom| {
            let k = (atom.residue.seq_num, atom.residue.ins_code);
            for r in ranges {
                if r.chain != atom.residue.chain {
                    continue;
                }
                let ks = (r.start_seq, r.start_ins);
                let ke = (r.end_seq, r.end_ins);
                if k >= ks && k <= ke {
                    return r.ss;
                }
            }
            SecondaryStructure::Coil
        })
        .collect()
}
