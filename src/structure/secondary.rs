use super::atom::{Atom, SecondaryStructure};

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
pub fn assign_ss(atoms: &[Atom], ranges: &[SsRange]) -> Vec<SecondaryStructure> {
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
