pub mod object;

use indexmap::IndexMap;
use std::collections::HashMap;

use object::MolecularObject;

/// Reference to a specific atom: (object_name, atom_index_within_object)
pub type AtomRef = (String, usize);

#[derive(Debug, Default)]
pub struct Scene {
    pub objects: IndexMap<String, MolecularObject>,
    /// Named selections → list of atom references
    pub selections: HashMap<String, Vec<AtomRef>>,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_object(&mut self, obj: MolecularObject) {
        self.objects.insert(obj.name.clone(), obj);
    }

    pub fn get(&self, name: &str) -> Option<&MolecularObject> {
        self.objects.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut MolecularObject> {
        self.objects.get_mut(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<MolecularObject> {
        self.objects.shift_remove(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &MolecularObject)> {
        self.objects.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut MolecularObject)> {
        self.objects.iter_mut()
    }

    /// Look up a named selection; "all" resolves to all atoms.
    pub fn resolve_selection(&self, name: &str) -> Vec<AtomRef> {
        if name == "all" || name == "*" {
            return self.all_atoms();
        }
        self.selections.get(name).cloned().unwrap_or_default()
    }

    /// All atoms across all objects.
    pub fn all_atoms(&self) -> Vec<AtomRef> {
        self.objects
            .iter()
            .flat_map(|(name, obj)| {
                (0..obj.structure.atoms.len()).map(move |i| (name.clone(), i))
            })
            .collect()
    }
}
