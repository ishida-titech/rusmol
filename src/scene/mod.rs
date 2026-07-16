pub mod object;

use indexmap::IndexMap;
use std::collections::HashMap;

use object::MolecularObject;

/// Reference to a specific atom: (object_name, atom_index_within_object)
pub type AtomRef = (String, usize);

// ── SceneDirty: fine-grained dirty flags for upload_scene ───────────────────

/// Bitmask indicating which parts of the GPU scene data need re-uploading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneDirty(u8);

impl SceneDirty {
    pub const NONE:    SceneDirty = SceneDirty(0);
    /// Spheres, cylinders, backbone, lines, ghost spheres.
    pub const ATOMS:   SceneDirty = SceneDirty(0b001);
    /// Ribbon mesh (also includes gap dashed cylinders → implies ATOMS rebuild
    /// for cylinder buffer, but the caller handles that).
    pub const RIBBON:  SceneDirty = SceneDirty(0b010);
    /// Surface mesh (most expensive).
    pub const SURFACE: SceneDirty = SceneDirty(0b100);
    /// All parts.
    pub const ALL:     SceneDirty = SceneDirty(0b111);

    #[inline] pub fn is_empty(self) -> bool  { self.0 == 0 }
    #[inline] pub fn contains(self, other: SceneDirty) -> bool { self.0 & other.0 == other.0 }
}

impl std::ops::BitOr for SceneDirty {
    type Output = Self;
    #[inline] fn bitor(self, rhs: Self) -> Self { SceneDirty(self.0 | rhs.0) }
}

impl std::ops::BitOrAssign for SceneDirty {
    #[inline] fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
}

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
