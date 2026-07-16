# rusmol v0.3 User Manual

rusmol is a lightweight molecular structure viewer for proteins and nucleic acids, written in Rust with wgpu (Metal backend). It features a PyMOL-compatible command set, interactive 3D rendering, and sub-second cold start.

## Table of Contents

- [Getting Started](#getting-started)
- [Command-Line Options](#command-line-options)
- [Mouse Controls](#mouse-controls)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [GUI Toolbar](#gui-toolbar)
- [Commands](#commands)
- [Selection Language](#selection-language)
- [Representations](#representations)
- [Color Specification](#color-specification)
- [Settings (`set`)](#settings-set)
- [Picking](#picking)
- [Default Display](#default-display)
- [Surface Types](#surface-types)
- [File Format Support](#file-format-support)

---

## Getting Started

```
rusmol file.pdb
rusmol file1.pdb file2.pdb
rusmol -c "show surface; color chain" file.pdb
```

rusmol loads PDB files, displays them in a 3D window, and presents an interactive prompt (`rusmol>`) in the terminal for entering commands.

---

## Command-Line Options

```
rusmol [OPTIONS] <FILES>...
```

| Option | Short | Description |
|--------|-------|-------------|
| `<FILES>...` | | PDB files to load (required, one or more) |
| `--command <CMD>` | `-c` | Commands to execute on startup (`;` separated) |
| `--verbose` | `-v` | Enable verbose logging |
| `--version` | | Print version |
| `--help` | | Print help |

### Examples

```
rusmol 1crn.pdb
rusmol -v 2je5.pdb
rusmol -c "show surface; color chain" 1abc.pdb
rusmol -c "hide ribbon; show lines" file1.pdb file2.pdb
```

---

## Mouse Controls

| Action | Effect |
|--------|--------|
| Left drag | Arcball rotation |
| Right drag | Pan (translate) |
| Scroll wheel | Zoom in/out |
| Left click (< 5 px movement) | Pick atom/residue |

Mouse events on the bottom toolbar are handled by the GUI and do not affect the 3D view.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Escape | Quit |

---

## GUI Toolbar

A toolbar at the bottom of the window provides preset view buttons:

| Button | Description |
|--------|-------------|
| **Default** | Ribbon (SS color) + ligand ball-and-stick (CPK) |
| **Chain Surface** | Surface (chain color) + ligand ball-and-stick (CPK) |
| **Binding Site** | Ligand ball-and-stick (CPK) + nearby protein residues within 4 A as sticks (darkened CPK) + remaining ribbon (light gray). Ligand–protein hydrogen bonds are drawn as yellow dashed lines. Hydrogens are hidden. |
| **Pocket Surface** | Ligand ball-and-stick + nearby protein surface (6 A, CPK, semi-transparent) + remaining ribbon (light gray) |

---

## Commands

### load

Load a PDB file into the scene.

```
load <path> [, name]
```

- `path` -- file path (required)
- `name` -- object name (default: filename stem)

```
load 1abc.pdb
load 1abc.pdb, myprotein
```

### select / sel

Create a named selection set.

```
select [name,] <expression>
sel    [name,] <expression>
```

- `name` -- selection name (default: `sele`)
- `expression` -- selection expression (see [Selection Language](#selection-language))

```
select active_site, chain A and resi 40-60
sel ligands, hetatm and not resn HOH
```

### show

Show a representation for selected atoms.

```
show <repr> [, selection]
```

- `repr` -- representation name (see [Representations](#representations))
- `selection` -- selection name or expression (default: all non-water atoms)

```
show ribbon
show surface, chain A
show ball_stick, hetatm
```

### hide

Hide a representation for selected atoms.

```
hide <repr> [, selection]
```

```
hide ribbon
hide surface, chain B
```

### color / colour

Set the color of selected atoms.

```
color <spec> [, selection]
colour <spec> [, selection]
```

- `spec` -- color specification (see [Color Specification](#color-specification))

```
color element
color red, chain A
color spectrum
color ss, chain A and not hetatm
```

### enable / disable

Toggle visibility of an entire object.

```
enable <object_name>
disable <object_name>
```

### delete / del

Remove an object from the scene.

```
delete <object_name>
del <object_name>
```

### zoom / z

Fit the camera to the selected atoms.

```
zoom [selection]
z    [selection]
```

- With no argument, fits to all atoms.

```
zoom
zoom chain A
z hetatm
```

### reset

Reset the camera to the default position (centered on all atoms, no rotation).

```
reset
```

### bg / background / bgcolor

Change the background color.

```
bg <color_name>
background <color_name>
bgcolor <color_name>
```

- Default background is black.
- Accepts any named color (see [Named Colors](#named-colors)).

```
bg white
bg black
background marine
```

### light

Adjust the directional light.

```
light [intensity <value>] [elevation <degrees>] [azimuth <degrees>]
```

At least one parameter must be specified. Short forms: `i` for intensity, `el` for elevation, `az` for azimuth.

```
light intensity 1.5
light el 45 az 30
light i 2.0 elevation 60 azimuth -30
```

| Parameter | Range | Default |
|-----------|-------|---------|
| intensity | >= 0.0 | 1.0 |
| elevation | -90 to 90 degrees | 30.0 |
| azimuth | any (degrees) | 20.0 |

### set

Adjust rendering and surface parameters.

```
set <name>, <value>
```

See [Settings](#settings-set) for all available parameters.

```
set transparency, 0.5
set surface_type, ses
set surface_quality, 0.3
set roughness, 0.6
```

### help / h / ?

Print the command reference.

```
help
```

### quit / q / exit

Exit the application.

```
quit
```

---

## Selection Language

Selections identify subsets of atoms for `show`, `hide`, `color`, `zoom`, and `select` commands.

### Primitives

| Keyword | Alias | Argument | Description |
|---------|-------|----------|-------------|
| `all` | `*` | -- | All atoms |
| `hetatm` | | -- | HETATM atoms only |
| `name` | | atom name | Atom name (e.g. `name CA`, `name OG1`) |
| `resn` | `resname` | residue name | Residue name (e.g. `resn ALA`, `resn HOH`) |
| `resi` | `resi_num`, `resnum` | range spec | Residue number (see below) |
| `chain` | | chain ID | Chain identifier (e.g. `chain A`) |
| `elem` | `element` | element symbol | Element (e.g. `elem C`, `elem FE`) |
| *object_name* | | -- | Filter by object name |

### Residue Number Syntax (`resi`)

| Form | Example | Meaning |
|------|---------|---------|
| Single | `resi 42` | Residue 42 |
| Range | `resi 1-100` | Residues 1 through 100 |
| List | `resi 1+5+10` | Residues 1, 5, and 10 |
| Mixed | `resi 1-10+15+20-30` | Residues 1-10, 15, and 20-30 |

### Logical Operators

| Operator | Precedence | Description |
|----------|------------|-------------|
| `not` | Highest | Negation |
| `and` | Medium | Intersection |
| `or` | Lowest | Union |
| `( )` | -- | Grouping |

### Examples

```
chain A
chain A and resn ALA
resi 1-50 or (hetatm and not resn HOH)
(chain A or chain B) and elem C
not hetatm
```

---

## Representations

| Name | Aliases | Description |
|------|---------|-------------|
| `ball_stick` | `ball-stick`, `bs`, `ball_and_stick`, `spheres` | Ball-and-stick model with half-bond coloring |
| `stick` | `sticks` | Uniform-radius sticks (thicker than ball-and-stick bonds); atoms are rounded to the bond radius, not van-der-Waals-scaled |
| `backbone` | `trace`, `ca_trace`, `ca` | C-alpha backbone trace with tubes and joint spheres |
| `ribbon` | `cartoon` | Ribbon/cartoon with secondary structure shapes |
| `surface` | | Molecular surface (Gaussian or SES) |
| `lines` | `line`, `wire` | Thin wireframe bonds |

When both Ribbon and BallAndStick are active on the same object, BallAndStick only draws non-water HETATM atoms and their bonds (to avoid visual clutter over the ribbon).

---

## Color Specification

### Color Schemes

| Scheme | Aliases | Description |
|--------|---------|-------------|
| `element` | `cpk` | CPK element colors |
| `chain` | `chainbows` | Color by chain (8-color cycle) |
| `ss` | `secondary`, `secondary_structure` | Secondary structure (helix=red, sheet=yellow, coil=light gray) |
| `spectrum` | `rainbow` | N-to-C rainbow gradient per chain (blue to red) |
| `b` | `b_factor`, `bfactor`, `tempfactor` | B-factor gradient (blue=low, white=mid, red=high) |

### Named Colors

| Color | RGB | Color | RGB |
|-------|-----|-------|-----|
| `red` | (1.00, 0.12, 0.12) | `pink` | (1.00, 0.75, 0.80) |
| `green` | (0.13, 0.70, 0.13) | `salmon` | (0.98, 0.50, 0.45) |
| `blue` | (0.12, 0.47, 0.71) | `wheat` | (0.96, 0.87, 0.70) |
| `white` | (1.00, 1.00, 1.00) | `teal` | (0.00, 0.50, 0.50) |
| `black` | (0.00, 0.00, 0.00) | `marine` | (0.00, 0.45, 1.00) |
| `yellow` | (1.00, 1.00, 0.00) | `forest` | (0.13, 0.55, 0.13) |
| `orange` | (1.00, 0.55, 0.00) | `limon` | (0.75, 1.00, 0.25) |
| `purple` / `violet` | (0.55, 0.00, 0.55) | `grey` / `gray` | (0.50, 0.50, 0.50) |
| `cyan` | (0.00, 0.80, 0.80) | `magenta` | (1.00, 0.00, 1.00) |

### CPK Element Colors

| Element | Color | Element | Color |
|---------|-------|---------|-------|
| H | White | FE | Orange-brown |
| C | Gray | ZN | Blue-gray |
| N | Blue | MG | Yellow-green |
| O | Red | CA | Green |
| S | Yellow | MN | Purple |
| P | Orange | CU | Brown |
| F, CL | Green | BR | Dark red |
| I | Purple | Unknown | Hot pink |

### Chain Colors (8-color cycle)

Green, Blue, Red, Orange, Purple, Brown, Pink, Gray

### Secondary Structure Colors

| Type | Color |
|------|-------|
| Helix | Red |
| Sheet | Yellow |
| Coil | Light gray |

---

## Settings (`set`)

```
set <name>, <value>
```

### Rendering Parameters

| Setting | Aliases | Range | Default | Description |
|---------|---------|-------|---------|-------------|
| `transparency` | `surface_transparency` | 0.0 - 1.0 | 0.0 | Surface transparency (0=opaque, 1=invisible) |
| `edge_strength` | | >= 0.0 | 1.0 | Edge outline strength (0=off) |
| `roughness` | | 0.0 - 1.0 | 0.4 | PBR roughness (0=mirror, 1=matte) |
| `metallic` | | 0.0 - 1.0 | 0.0 | PBR metallic factor |
| `ibl_intensity` | | >= 0.0 | 1.0 | Environment light (IBL) intensity |
| `shadow_strength` | `shadow` | 0.0 - 1.0 | 0.4 | Shadow darkness (0=no shadow) |
| `bloom_threshold` | | >= 0.0 | 1.0 | Bloom luminance threshold |
| `bloom_intensity` | `bloom` | >= 0.0 | 0.0 | Bloom glow intensity (0=off) |

### Surface Parameters

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| `surface_type` | `gaussian`, `ses` | `gaussian` | Surface algorithm |
| `surface_quality` | 0.2 - 2.0 (A) | 0.5 | Grid step size (smaller=finer) |

`surface_type` also accepts `connolly` and `molecular` as aliases for `ses`.

### Per-Object Color Overrides

```
set surface_color, <color> [, object_name]
set cartoon_color, <color> [, object_name]
```

- `cartoon_color` alias: `ribbon_color`
- Use `default` to restore per-atom coloring.

```
set surface_color, blue
set cartoon_color, default, myprotein
```

---

## Picking

Click on the 3D view (left click with < 5 px movement) to identify atoms and residues.

| Target | Result | Terminal Output |
|--------|--------|-----------------|
| Ball-and-stick atom | Exact atom hit | `Picked: ALA A:42  CA` |
| Ribbon / Surface | Nearest residue (ghost-sphere search, 20 px radius) | `Picked: ALA A:42` |
| Background | Clear highlight | (none) |

The picked residue is highlighted with an orange rim glow effect.

---

## Default Display

When a PDB file is loaded, the following defaults are applied:

| Atom Category | Representation | Color |
|---------------|----------------|-------|
| Polymer (protein/nucleic acid) | Ribbon | Secondary structure colors |
| Ligand (non-water HETATM) | Ball-and-stick | CPK element colors |
| Water (HOH / WAT / DOD) | Hidden | -- |

---

## Surface Types

### Gaussian Surface (default)

A smooth density-based surface computed from atom VdW radii with Gaussian blurring. Fast to compute and produces visually smooth surfaces.

### SES (Solvent-Excluded Surface)

The molecular surface that excludes the solvent, computed via Euclidean Distance Transform (EDT) with a probe radius of 1.4 A (water molecule). Shows the actual molecular boundary that solvent molecules cannot penetrate. Also known as Connolly surface.

Switch between surface types:

```
set surface_type, gaussian
set surface_type, ses
```

Control mesh quality (grid resolution in Angstroms):

```
set surface_quality, 0.5    # default, balanced
set surface_quality, 0.3    # finer mesh, slower
set surface_quality, 1.0    # coarser mesh, faster
```

---

## File Format Support

### PDB Format

rusmol reads the following PDB record types:

| Record | Usage |
|--------|-------|
| ATOM | Standard residue atom coordinates |
| HETATM | Ligand, modified residue, and water coordinates |
| HELIX | Alpha helix secondary structure |
| SHEET | Beta sheet secondary structure |
| CONECT | Explicit bond connectivity |
| COMPND | Molecule names per chain |
| SEQRES | Polymer sequence (used for polymer/ligand classification) |
| HETNAM | Chemical name of HET compounds |
| HETSYN | Synonyms of HET compounds |
| END / ENDMDL | End of model (only the first model is read) |

### Bond Estimation

Bonds are determined from:

1. **Built-in tables** -- Standard amino acids, common modified residues, small molecules, and single-atom ions (no network access required)
2. **RCSB CCD download** -- For unknown residues, the Chemical Component Dictionary is fetched from RCSB and cached in `~/.cache/rusmol/ccd/`
3. **CONECT records** -- Explicit bonds from the PDB file are added (with deduplication)
4. **Peptide bonds** -- Inferred from residue sequence continuity (C-N distance validated)
5. **Disulfide bonds** -- CYS/CYX SG-SG pairs within 2.3 A
6. **Nucleic acid backbone** -- O3'-P bonds within 1.7 A
7. **Hydrogens** -- Hydrogens are absent from the bond tables, so any H present in the file (e.g. PDBQT polar hydrogens) is connected to its nearest heavy atom within the same residue (<= 1.3 A) so it does not float

### Loading Information

When a file is loaded, rusmol prints a summary to the terminal:

```
Loaded 'myprotein' from path/to/file.pdb
Chains:
  A  Protein       1-250  EXAMPLE PROTEIN
  B  Protein       1-250  EXAMPLE PROTEIN
Ligands:
  ATP     A:301  ADENOSINE-5'-TRIPHOSPHATE
  MG      A:302  MAGNESIUM ION
```
