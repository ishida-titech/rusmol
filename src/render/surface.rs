use crate::render::ribbon::RibbonVertex;
use crate::scene::object::REP_SURFACE;
use crate::structure::atom::Structure;
use crate::util::color::vdw_radius;
use glam::Vec3;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

const SIGMA: f32 = 1.2;
const THRESHOLD: f32 = 0.5;
const MARGIN: f32 = 3.0;
const CUTOFF: f32 = 5.0;

/// Probe radius for SES (water molecule radius).
const PROBE_RADIUS: f32 = 1.4;

/// Morphological-closing radius (Å) applied to the density field before
/// Marching Cubes. Closing (dilate then erode) seals interior tunnels and
/// surface pits narrower than 2× this radius — including the spurious
/// through-holes Marching Cubes can leave in thin protein regions — while
/// leaving genuine clefts and pockets wider than the ball untouched.
const SURFACE_CLOSE_RADIUS: f32 = 1.5;

/// Taubin mesh-smoothing parameters (removes marching-cubes grid quantization
/// bumps without the volumetric shrinkage of plain Laplacian smoothing).
/// The iteration count is passed in at runtime (`set surface_smooth`).
const SMOOTH_LAMBDA: f32 = 0.5;
const SMOOTH_MU: f32 = -0.53;

/// Largest boundary loop (in edges) that `fill_all_holes` will patch. Marching
/// Cubes leaves only a handful of tiny gaps, so this stays small; a larger loop
/// is left alone rather than fan-filled into a flat sheet.
const HOLE_MAX_EDGES: usize = 64;

/// Surface computation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceType {
    /// Gaussian density isosurface (default).
    Gaussian,
    /// Solvent-Excluded Surface via distance transform.
    Ses,
}

// ── Marching Cubes lookup tables (Paul Bourke convention) ─────────────────────
// Corner numbering: 0=(i,j,k), 1=(i+1,j,k), 2=(i+1,j+1,k), 3=(i,j+1,k),
//                   4=(i,j,k+1), 5=(i+1,j,k+1), 6=(i+1,j+1,k+1), 7=(i,j+1,k+1)
// Edge numbering:   0:(0-1), 1:(1-2), 2:(2-3), 3:(3-0),
//                   4:(4-5), 5:(5-6), 6:(6-7), 7:(7-4),
//                   8:(0-4), 9:(1-5), 10:(2-6), 11:(3-7)

#[allow(clippy::unreadable_literal)]
const EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

const TRI_TABLE: [[i8; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1],
    [3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1],
    [3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1],
    [9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1],
    [8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1],
    [3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1],
    [1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1],
    [4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1],
    [4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1],
    [2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1],
    [9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1],
    [10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1],
    [5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1],
    [5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1],
    [0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1],
    [1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1],
    [8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1],
    [2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1],
    [7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1],
    [9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1],
    [2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1],
    [11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1],
    [9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1],
    [5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1],
    [11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1],
    [11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1],
    [9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1],
    [5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1],
    [2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1],
    [6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1],
    [3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1],
    [6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1],
    [6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1],
    [1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1],
    [8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1],
    [7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1],
    [3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1],
    [0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1],
    [9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1],
    [8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1],
    [5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1],
    [0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1],
    [6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1],
    [10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1],
    [10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1],
    [8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1],
    [1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1],
    [0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1],
    [10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1],
    [3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1],
    [6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1],
    [9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1],
    [8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1],
    [3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1],
    [6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1],
    [10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1],
    [10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1],
    [1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1],
    [2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1],
    [7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1],
    [7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1],
    [2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1],
    [1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1],
    [11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1],
    [8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1],
    [0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1],
    [7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1],
    [7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1],
    [2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1],
    [1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1],
    [10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1],
    [10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1],
    [0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1],
    [7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1],
    [6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1],
    [9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1],
    [6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1],
    [4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1],
    [10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1],
    [8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1],
    [0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1],
    [1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1],
    [10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1],
    [4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1],
    [10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1],
    [9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1],
    [7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1],
    [3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1],
    [7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1],
    [3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1],
    [6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1],
    [9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1],
    [1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1],
    [4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1],
    [7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1],
    [6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1],
    [0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1],
    [6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1],
    [0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1],
    [11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1],
    [6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1],
    [5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1],
    [9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1],
    [1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1],
    [1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1],
    [10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1],
    [0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1],
    [5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1],
    [10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1],
    [11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1],
    [9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1],
    [7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1],
    [2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1],
    [8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1],
    [9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1],
    [9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1],
    [1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1],
    [9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1],
    [9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1],
    [5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1],
    [0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1],
    [10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1],
    [2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1],
    [0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1],
    [0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1],
    [9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1],
    [5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1],
    [3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1],
    [5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1],
    [8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1],
    [9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1],
    [1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1],
    [3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1],
    [4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1],
    [9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1],
    [11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1],
    [11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1],
    [2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1],
    [9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1],
    [3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1],
    [1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1],
    [4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1],
    [4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1],
    [3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1],
    [0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1],
    [9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1],
    [1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];

// Edge endpoint corner pairs (corner indices 0-7)
const EDGE_CORNERS: [(usize, usize); 12] = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
];

// Corner offsets from (i,j,k)
const CORNER_OFF: [(i32,i32,i32); 8] = [
    (0,0,0),(1,0,0),(1,1,0),(0,1,0),
    (0,0,1),(1,0,1),(1,1,1),(0,1,1),
];

// ── Density field computation ─────────────────────────────────────────────────

/// Fill density grid using Gaussian blobs (original method).
/// atoms_data: (position, color, residue_id, vdw_radius)
fn fill_density_gaussian(
    atoms_data: &[(Vec3, [f32; 3], u32, f32)],
    density: &mut [f32],
    color_sum: &mut [[f32; 3]],
    min: Vec3,
    nx: usize,
    ny: usize,
    nz: usize,
    step: f32,
) {
    let inv_2s2 = 1.0 / (2.0 * SIGMA * SIGMA);
    let cutoff2 = CUTOFF * CUTOFF;
    let cells_r = (CUTOFF / step).ceil() as i32 + 1;

    for (pos, color, _, _) in atoms_data {
        let fc = (*pos - min) / step;
        let ci = fc.x as i32;
        let cj = fc.y as i32;
        let ck = fc.z as i32;

        for di in -cells_r..=cells_r {
            let ii = ci + di;
            if ii < 0 || ii >= nx as i32 { continue; }
            let cx = min.x + (ii as f32) * step;
            let dxf = cx - pos.x;
            if dxf * dxf > cutoff2 { continue; }

            for dj in -cells_r..=cells_r {
                let jj = cj + dj;
                if jj < 0 || jj >= ny as i32 { continue; }
                let cy = min.y + (jj as f32) * step;
                let dyf = cy - pos.y;
                if dxf * dxf + dyf * dyf > cutoff2 { continue; }

                for dk in -cells_r..=cells_r {
                    let kk = ck + dk;
                    if kk < 0 || kk >= nz as i32 { continue; }
                    let cz = min.z + (kk as f32) * step;
                    let dzf = cz - pos.z;
                    let r2 = dxf * dxf + dyf * dyf + dzf * dzf;
                    if r2 > cutoff2 { continue; }

                    let w = (-r2 * inv_2s2).exp();
                    let idx = (ii as usize) * ny * nz + (jj as usize) * nz + (kk as usize);
                    density[idx] += w;
                    color_sum[idx][0] += color[0] * w;
                    color_sum[idx][1] += color[1] * w;
                    color_sum[idx][2] += color[2] * w;
                }
            }
        }
    }
}

/// Compute 1D squared Euclidean distance transform in-place.
/// Input: f[q] = 0.0 if in set, large value otherwise.
/// Output: f[q] = min_p( (q - p)^2 + f_input[p] ), i.e. squared distance in grid units
/// to the nearest set point (Felzenszwalb & Huttenlocher, 2012).
fn edt_1d(f: &mut [f32]) {
    let n = f.len();
    if n <= 1 { return; }

    let mut v = vec![0usize; n];   // parabola locations
    let mut z = vec![0.0f32; n + 1]; // boundaries
    let mut d = vec![0.0f32; n];

    let mut k: usize = 0;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        loop {
            let vk = v[k];
            let s = ((f[q] + (q * q) as f32) - (f[vk] + (vk * vk) as f32))
                / (2 * q - 2 * vk) as f32;
            if k > 0 && s <= z[k] {
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
        }
    }

    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let dq = q as f32 - v[k] as f32;
        d[q] = dq * dq + f[v[k]];
    }
    f[..n].copy_from_slice(&d);
}

/// 3D squared Euclidean Distance Transform (separable, O(n)).
/// Each pass is parallelised over independent scanlines using rayon.
fn edt_3d(grid: &mut [f32], nx: usize, ny: usize, nz: usize) {
    // Pass 1: along z — each (i,j) row is independent and contiguous
    grid.par_chunks_mut(nz).for_each(|row| {
        edt_1d(row);
    });

    // Pass 2: along y — each (i, k) column is independent (stride = nz)
    // Process per i-slab in parallel; within each slab iterate over k sequentially.
    let slab = ny * nz;
    grid.par_chunks_mut(slab).for_each(|plane| {
        let mut buf = vec![0.0f32; ny];
        for k in 0..nz {
            for j in 0..ny {
                buf[j] = plane[j * nz + k];
            }
            edt_1d(&mut buf);
            for j in 0..ny {
                plane[j * nz + k] = buf[j];
            }
        }
    });

    // Pass 3: along x — each (j, k) column is independent (stride = ny*nz)
    // Transpose the grid into x-major order, apply EDT per column, transpose back.
    let slab = ny * nz;
    let mut transposed = vec![0.0f32; nx * ny * nz];
    // grid[i*slab + j*nz + k] → transposed[(j*nz + k)*nx + i]
    for i in 0..nx {
        for jk in 0..slab {
            transposed[jk * nx + i] = grid[i * slab + jk];
        }
    }
    // Each row of transposed (length nx) is an independent x-column
    transposed.par_chunks_mut(nx).for_each(|row| {
        edt_1d(row);
    });
    // Transpose back
    for i in 0..nx {
        for jk in 0..slab {
            grid[i * slab + jk] = transposed[jk * nx + i];
        }
    }
}

/// Fill density grid for Solvent-Excluded Surface via EDT.
///
/// Algorithm (correct SES, not SAS):
/// 1. Compute signed distance to nearest VdW surface for each grid point.
/// 2. Mark "accessible" region (probe center can sit here): d_vdw >= r_probe.
/// 3. 3D EDT: squared distance from each point to nearest accessible point.
/// 4. SES isosurface: d_to_accessible == r_probe.
///    Points where d_to_accessible >= r_probe are INSIDE the SES.
fn fill_density_ses(
    atoms_data: &[(Vec3, [f32; 3], u32, f32)],
    density: &mut [f32],
    color_sum: &mut [[f32; 3]],
    min: Vec3,
    nx: usize,
    ny: usize,
    nz: usize,
    step: f32,
) {
    let n = nx * ny * nz;

    // ── Step 1: min distance to VdW surface + closest atom ──────────────────
    // Parallelise over i-slabs; each slab owns its own slice of the arrays.
    let slab = ny * nz;
    let mut min_vdw_dist = vec![f32::MAX; n];
    let mut closest_atom = vec![0usize; n];

    min_vdw_dist
        .par_chunks_mut(slab)
        .zip(closest_atom.par_chunks_mut(slab))
        .enumerate()
        .for_each(|(ii, (dist_slab, atom_slab))| {
            let cx_base = min.x + (ii as f32) * step;
            for (atom_idx, (pos, _, _, vdw_r)) in atoms_data.iter().enumerate() {
                let max_r = vdw_r + PROBE_RADIUS + 2.0;
                let dxf = cx_base - pos.x;
                if dxf.abs() > max_r { continue; }

                let cells_r = (max_r / step).ceil() as i32 + 1;
                let fc = (*pos - min) / step;
                let cj_center = fc.y as i32;
                let ck_center = fc.z as i32;

                for dj in -cells_r..=cells_r {
                    let jj = cj_center + dj;
                    if jj < 0 || jj >= ny as i32 { continue; }
                    let cy = min.y + (jj as f32) * step;
                    let dyf = cy - pos.y;
                    if dxf * dxf + dyf * dyf > max_r * max_r { continue; }

                    for dk in -cells_r..=cells_r {
                        let kk = ck_center + dk;
                        if kk < 0 || kk >= nz as i32 { continue; }
                        let cz = min.z + (kk as f32) * step;
                        let dzf = cz - pos.z;
                        let r = (dxf * dxf + dyf * dyf + dzf * dzf).sqrt();
                        let dist = r - vdw_r;

                        let local_idx = (jj as usize) * nz + (kk as usize);
                        if dist < dist_slab[local_idx] {
                            dist_slab[local_idx] = dist;
                            atom_slab[local_idx] = atom_idx;
                        }
                    }
                }
            }
        });

    // ── Step 2: Binary accessibility → EDT input ────────────────────────────
    // Accessible = probe center can sit here (d_vdw >= r_probe).
    // Points far from all atoms (min_vdw_dist = MAX) are also accessible.
    let big = (nx * nx + ny * ny + nz * nz) as f32 + 1.0;
    let mut dist_sq: Vec<f32> = min_vdw_dist
        .par_iter()
        .map(|&d| if d >= PROBE_RADIUS { 0.0 } else { big })
        .collect();
    drop(min_vdw_dist); // no longer needed, free memory before EDT

    // ── Step 3: 3D EDT → squared distance to nearest accessible point ───────
    edt_3d(&mut dist_sq, nx, ny, nz);

    // ── Step 4: Convert to density for Marching Cubes ───────────────────────
    // d_to_accessible = sqrt(dist_sq) * step  (in Å)
    // Inside SES:  d_to_accessible >= PROBE_RADIUS
    // SES surface: d_to_accessible == PROBE_RADIUS
    // density = (d_to_accessible - PROBE_RADIUS) + THRESHOLD
    //         = THRESHOLD on the surface, > THRESHOLD inside, < THRESHOLD outside
    density
        .par_iter_mut()
        .zip(color_sum.par_iter_mut())
        .zip(dist_sq.par_iter().zip(closest_atom.par_iter()))
        .for_each(|((den, col), (&dsq, &aidx))| {
            let d_ang = dsq.sqrt() * step;
            let val = (d_ang - PROBE_RADIUS) + THRESHOLD;
            if val > 0.0 {
                *den = val;
                let (_, color, _, _) = atoms_data[aidx];
                *col = [color[0] * val, color[1] * val, color[2] * val];
            }
        });
}

/// Separable 3×3×3 box blur, applied `passes` times in place. Border cells
/// clamp (the missing neighbour repeats the cell's own value). Used to mollify
/// the closing set's binary indicator so its 0.5-isosurface is smooth rather
/// than following the grid staircase.
fn box_blur3(grid: &mut [f32], nx: usize, ny: usize, nz: usize, passes: usize) {
    let slab = ny * nz;
    let third = 1.0 / 3.0;
    for _ in 0..passes {
        // z axis: each (i,j) row is contiguous.
        grid.par_chunks_mut(nz).for_each(|row| {
            let n = row.len();
            if n == 0 { return; }
            let mut prev = row[0]; // original value of the previous cell
            for i in 0..n {
                let cur = row[i];
                let nxt = if i + 1 < n { row[i + 1] } else { cur };
                row[i] = (prev + cur + nxt) * third;
                prev = cur;
            }
        });
        // y axis: gather each (i,k) column into a scratch buffer.
        grid.par_chunks_mut(slab).for_each(|plane| {
            let mut col = vec![0.0f32; ny];
            for k in 0..nz {
                for j in 0..ny {
                    col[j] = plane[j * nz + k];
                }
                let mut prev = col[0];
                for j in 0..ny {
                    let cur = col[j];
                    let nxt = if j + 1 < ny { col[j + 1] } else { cur };
                    plane[j * nz + k] = (prev + cur + nxt) * third;
                    prev = cur;
                }
            }
        });
        // x axis: transpose to x-major, blur contiguous rows, transpose back.
        let mut t = vec![0.0f32; nx * ny * nz];
        for i in 0..nx {
            for jk in 0..slab {
                t[jk * nx + i] = grid[i * slab + jk];
            }
        }
        t.par_chunks_mut(nx).for_each(|row| {
            let n = row.len();
            if n == 0 { return; }
            let mut prev = row[0];
            for i in 0..n {
                let cur = row[i];
                let nxt = if i + 1 < n { row[i + 1] } else { cur };
                row[i] = (prev + cur + nxt) * third;
                prev = cur;
            }
        });
        for i in 0..nx {
            for jk in 0..slab {
                grid[i * slab + jk] = t[jk * nx + i];
            }
        }
    }
}

/// Morphological closing of the inside set (density >= THRESHOLD) on the
/// density grid: dilate by `radius` Å, then erode by the same radius. This
/// seals interior tunnels and pits narrower than 2×`radius` while preserving
/// the outer surface, because closing is extensive (E ⊇ A) and only ever
/// *raises* a cell's density here — never lowers it.
///
/// Both passes reuse `edt_3d`: the dilation is "within `radius` of an inside
/// cell", the erosion is "at least `radius` from a non-dilated cell". The
/// erosion's boundary is a thresholded set, so its raw indicator staircases
/// along the grid; we box-blur that indicator and take its 0.5-isosurface as
/// the sealed density, which gives Marching Cubes a smooth patch instead of
/// the blocky ridges a raw distance step would leave.
///
/// Returns the number of cells newly pushed to/above THRESHOLD (for logging).
fn close_density(
    density: &mut [f32],
    nx: usize,
    ny: usize,
    nz: usize,
    step: f32,
    radius: f32,
) -> usize {
    if radius <= 0.0 {
        return 0;
    }
    let r_cells = radius / step; // ball radius in grid cells
    let r_cells_sq = r_cells * r_cells;
    let big = (nx * nx + ny * ny + nz * nz) as f32 + 1.0;

    // ── Dilation: squared distance from each cell to the nearest inside cell.
    // Seeds (distance 0) = inside cells; everything else starts at `big`.
    let mut dist_a: Vec<f32> = density
        .par_iter()
        .map(|&d| if d >= THRESHOLD { 0.0 } else { big })
        .collect();
    edt_3d(&mut dist_a, nx, ny, nz);
    // Dilated set D = { x : dist_a[x] <= r_cells_sq }.

    // ── Erosion of D: squared distance from each cell to the nearest cell
    // OUTSIDE D. Seeds (distance 0) = complement(D); D cells start at `big`.
    let mut dist_d: Vec<f32> = dist_a
        .par_iter()
        .map(|&da| if da > r_cells_sq { 0.0 } else { big })
        .collect();
    drop(dist_a);
    edt_3d(&mut dist_d, nx, ny, nz);
    // Closed set E = erosion of D = { x : dist_d[x] > r_cells_sq }.

    // ── Mollify the binary indicator of E so its isosurface is smooth. ──
    let mut soft: Vec<f32> = dist_d
        .par_iter()
        .map(|&dd| if dd > r_cells_sq { 1.0 } else { 0.0 })
        .collect();
    drop(dist_d);
    box_blur3(&mut soft, nx, ny, nz, 2);

    // ── Combine: the field equals THRESHOLD where the smoothed indicator is
    // 0.5 (∂E), higher inside, lower outside. `max` keeps the finer original
    // density everywhere except in the newly-sealed tunnel/pit cells.
    density
        .par_iter_mut()
        .zip(soft.par_iter())
        .map(|(den, &s)| {
            let field = s * (2.0 * THRESHOLD); // == THRESHOLD at s = 0.5
            if field > *den {
                let was_outside = *den < THRESHOLD;
                *den = field;
                (was_outside && field >= THRESHOLD) as usize
            } else {
                0
            }
        })
        .sum::<usize>()
}

/// Fill in colours for cells the closing step sealed (they carry no atom
/// colour of their own) by propagating from their coloured neighbours. Sealed
/// tunnel cells are only a few cells thick, so a bounded flood from the known
/// frontier reaches them all in a handful of passes.
fn inpaint_colors(
    cell_color: &mut [[f32; 3]],
    colored: &[bool],
    density: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
) {
    let n = nx * ny * nz;
    let slab = ny * nz;
    let neigh = |i: usize| -> [Option<usize>; 6] {
        let k = i % nz;
        let j = (i / nz) % ny;
        let ii = i / slab;
        [
            if ii > 0 { Some(i - slab) } else { None },
            if ii + 1 < nx { Some(i + slab) } else { None },
            if j > 0 { Some(i - nz) } else { None },
            if j + 1 < ny { Some(i + nz) } else { None },
            if k > 0 { Some(i - 1) } else { None },
            if k + 1 < nz { Some(i + 1) } else { None },
        ]
    };

    // Cells that need a colour: any cell whose density the closing raised above
    // the normaliser's 1e-6 floor (so it no longer takes the neutral-grey
    // fallback) yet which carries no atom colour of its own. This must include
    // the sub-THRESHOLD "ramp" cells just outside ∂E, not only the sealed
    // interior — Marching Cubes interpolates vertex colours across the isolevel,
    // so a black ramp cell would bleed into the surface it borders.
    let targets: Vec<usize> = (0..n)
        .into_par_iter()
        .filter(|&i| density[i] > 1e-6 && !colored[i])
        .collect();
    if targets.is_empty() {
        return;
    }

    let mut known = colored.to_vec();
    for _ in 0..64 {
        // Snapshot-based pass: each target reads the previous frontier only.
        let updates: Vec<(usize, [f32; 3])> = targets
            .par_iter()
            .filter(|&&i| !known[i])
            .filter_map(|&i| {
                let mut sum = [0.0f32; 3];
                let mut cnt = 0u32;
                for nb in neigh(i).into_iter().flatten() {
                    if known[nb] {
                        sum[0] += cell_color[nb][0];
                        sum[1] += cell_color[nb][1];
                        sum[2] += cell_color[nb][2];
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    let inv = 1.0 / cnt as f32;
                    Some((i, [sum[0] * inv, sum[1] * inv, sum[2] * inv]))
                } else {
                    None
                }
            })
            .collect();
        if updates.is_empty() {
            break;
        }
        for (i, c) in updates {
            cell_color[i] = c;
            known[i] = true;
        }
    }
}

/// Build an isosurface mesh from the molecular density field.
/// Appends to the provided vertex and index buffers (same layout as ribbon).
/// Water molecules (HOH/WAT/DOD) are excluded from the density computation.
/// `residue_ids[i]` is the residue identifier for atom `i`.
/// May take 1-5 seconds for large structures.
pub fn build_surface(
    structure: &Structure,
    atom_colors: &[[f32; 3]],
    residue_ids: &[u32],
    atom_rep_show: &[u8],
    surface_type: SurfaceType,
    step: f32,
    smooth_iters: usize,
    vertices: &mut Vec<RibbonVertex>,
    indices: &mut Vec<u32>,
) {
    // ── 1. Collect atoms with REP_SURFACE bit set, grouped by chain ────────────
    let mut chain_atoms: HashMap<char, Vec<(Vec3, [f32; 3], u32, f32)>> = HashMap::new();
    for (i, a) in structure.atoms.iter().enumerate() {
        if atom_rep_show.get(i).copied().unwrap_or(0) & REP_SURFACE == 0 {
            continue;
        }
        if !structure.is_polymer_atom(a) {
            continue;
        }
        let r = vdw_radius(&a.element);
        chain_atoms
            .entry(a.residue.chain)
            .or_default()
            .push((a.position, atom_colors[i], residue_ids[i], r));
    }

    if chain_atoms.is_empty() {
        return;
    }

    // Sort chains for deterministic output order
    let mut chains: Vec<char> = chain_atoms.keys().copied().collect();
    chains.sort();

    for chain_id in chains {
        let atoms_data = &chain_atoms[&chain_id];
        build_surface_for_atoms(atoms_data, surface_type, step, smooth_iters, vertices, indices);
    }
}

/// Build the surface mesh for a single group of atoms (typically one chain).
fn build_surface_for_atoms(
    atoms_data: &[(Vec3, [f32; 3], u32, f32)],
    surface_type: SurfaceType,
    step: f32,
    smooth_iters: usize,
    vertices: &mut Vec<RibbonVertex>,
    indices: &mut Vec<u32>,
) {
    if atoms_data.is_empty() {
        return;
    }

    // Max VdW radius for bounding box margin
    let max_vdw = atoms_data.iter().map(|a| a.3).fold(0.0f32, f32::max);
    let margin = match surface_type {
        SurfaceType::Gaussian => MARGIN,
        SurfaceType::Ses => max_vdw + PROBE_RADIUS + 1.0,
    };

    // ── 2. Bounding box ────────────────────────────────────────────────────────
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for (pos, _, _, _) in atoms_data {
        min = min.min(*pos);
        max = max.max(*pos);
    }
    min -= Vec3::splat(margin);
    max += Vec3::splat(margin);

    // ── 3. Grid setup ──────────────────────────────────────────────────────────
    let nx = ((max.x - min.x) / step).ceil() as usize + 1;
    let ny = ((max.y - min.y) / step).ceil() as usize + 1;
    let nz = ((max.z - min.z) / step).ceil() as usize + 1;

    let n_cells = nx * ny * nz;
    if n_cells > 64_000_000 {
        // ~28-32 B/cell → 64M≈2GB, 128M≈4GB, 256M≈8GB
        log::warn!(
            "Surface grid {nx}×{ny}×{nz} = {}M cells (~{} MB peak)",
            n_cells / 1_000_000,
            n_cells / 1_000_000 * 32,
        );
    }

    let n = nx * ny * nz;
    let mut density: Vec<f32> = vec![0.0; n];
    let mut color_sum: Vec<[f32; 3]> = vec![[0.0; 3]; n];

    match surface_type {
        SurfaceType::Gaussian => {
            fill_density_gaussian(&atoms_data, &mut density, &mut color_sum, min, nx, ny, nz, step);
        }
        SurfaceType::Ses => {
            fill_density_ses(&atoms_data, &mut density, &mut color_sum, min, nx, ny, nz, step);
        }
    }

    // Cells that received an atom colour, captured before closing seals any
    // tunnels; the sealed cells carry no colour and are inpainted afterwards.
    let colored: Vec<bool> = density.par_iter().map(|&d| d > 1e-6).collect();

    // Morphological closing: seal spurious interior tunnels / narrow pits.
    let n_sealed = close_density(&mut density, nx, ny, nz, step, SURFACE_CLOSE_RADIUS);

    // Normalize per-cell colors
    let mut cell_color: Vec<[f32; 3]> = density
        .par_iter()
        .zip(color_sum.par_iter())
        .map(|(&d, c)| {
            if d > 1e-6 {
                let inv = 1.0 / d;
                [c[0] * inv, c[1] * inv, c[2] * inv]
            } else {
                [0.5, 0.5, 0.5]
            }
        })
        .collect();
    drop(color_sum); // free memory before MC

    // Give the sealed tunnel cells a colour borrowed from their neighbours so
    // they blend into the surface instead of rendering as black/grey patches.
    if n_sealed > 0 {
        inpaint_colors(&mut cell_color, &colored, &density, nx, ny, nz);
        log::debug!("close_density: sealed {n_sealed} tunnel/pit cells");
    }
    drop(colored);

    // ── 5. Marching Cubes (parallelised over the ci dimension) ────────────────
    // density and cell_color are read-only from here on; share via references.
    let density_ref: &[f32] = &density;
    let cell_color_ref: &[[f32; 3]] = &cell_color;

    // Helper: sample density, returning 0 outside the grid.
    let get_d = |d: &[f32], i: i32, j: i32, k: i32| -> f32 {
        if i < 0 || j < 0 || k < 0 || i >= nx as i32 || j >= ny as i32 || k >= nz as i32 {
            0.0
        } else {
            d[(i as usize) * ny * nz + (j as usize) * nz + (k as usize)]
        }
    };

    // Each ci slice produces its own (verts, idxs); merge afterwards.
    let slices: Vec<(Vec<RibbonVertex>, Vec<u32>)> = (0..nx - 1)
        .into_par_iter()
        .map(|ci| {
            let mut local_verts: Vec<RibbonVertex> = Vec::new();
            let mut local_idxs: Vec<u32> = Vec::new();

            for cj in 0..ny - 1 {
                for ck in 0..nz - 1 {
                    // Densities and flat indices at the 8 cube corners
                    let mut cd = [0.0f32; 8];
                    let mut ci_flat = [0usize; 8];
                    for c in 0..8 {
                        let (di, dj, dk) = CORNER_OFF[c];
                        let ii = ci as i32 + di;
                        let jj = cj as i32 + dj;
                        let kk = ck as i32 + dk;
                        let flat = (ii as usize) * ny * nz + (jj as usize) * nz + (kk as usize);
                        cd[c] = density_ref[flat];
                        ci_flat[c] = flat;
                    }

                    // Cube index
                    let mut cube_idx = 0u8;
                    for c in 0..8 {
                        if cd[c] >= THRESHOLD {
                            cube_idx |= 1 << c;
                        }
                    }

                    let edge_mask = EDGE_TABLE[cube_idx as usize];
                    if edge_mask == 0 { continue; }

                    // Compute a vertex for each intersected edge
                    // Indices are local to this slice's vertex list.
                    let mut edge_v = [0u32; 12];
                    for e in 0..12u32 {
                        if (edge_mask & (1 << e)) == 0 { continue; }

                        let (c0, c1) = EDGE_CORNERS[e as usize];
                        let d0 = cd[c0];
                        let d1 = cd[c1];

                        let t = if (d1 - d0).abs() > 1e-7 {
                            (THRESHOLD - d0) / (d1 - d0)
                        } else {
                            0.5
                        };
                        let t = t.clamp(0.0, 1.0);

                        let (di0, dj0, dk0) = CORNER_OFF[c0];
                        let (di1, dj1, dk1) = CORNER_OFF[c1];
                        let i0 = ci as i32 + di0;
                        let j0 = cj as i32 + dj0;
                        let k0 = ck as i32 + dk0;
                        let i1 = ci as i32 + di1;
                        let j1 = cj as i32 + dj1;
                        let k1 = ck as i32 + dk1;

                        let p0 = Vec3::new(
                            min.x + (i0 as f32) * step,
                            min.y + (j0 as f32) * step,
                            min.z + (k0 as f32) * step,
                        );
                        let p1 = Vec3::new(
                            min.x + (i1 as f32) * step,
                            min.y + (j1 as f32) * step,
                            min.z + (k1 as f32) * step,
                        );
                        let position = p0.lerp(p1, t).to_array();

                        // Normal via gradient (finite differences), interpolated
                        let grad = |ii: i32, jj: i32, kk: i32| -> [f32; 3] {
                            [
                                get_d(density_ref, ii+1,jj,kk) - get_d(density_ref, ii-1,jj,kk),
                                get_d(density_ref, ii,jj+1,kk) - get_d(density_ref, ii,jj-1,kk),
                                get_d(density_ref, ii,jj,kk+1) - get_d(density_ref, ii,jj,kk-1),
                            ]
                        };
                        let g0 = grad(i0, j0, k0);
                        let g1 = grad(i1, j1, k1);
                        let gx = -(g0[0] + t * (g1[0] - g0[0]));
                        let gy = -(g0[1] + t * (g1[1] - g0[1]));
                        let gz = -(g0[2] + t * (g1[2] - g0[2]));
                        let glen = (gx*gx + gy*gy + gz*gz).sqrt().max(1e-8);
                        let normal = [gx/glen, gy/glen, gz/glen];

                        // Color: interpolate between corner cells
                        let col0 = cell_color_ref[ci_flat[c0]];
                        let col1 = cell_color_ref[ci_flat[c1]];
                        let color = [
                            col0[0] + t * (col1[0] - col0[0]),
                            col0[1] + t * (col1[1] - col0[1]),
                            col0[2] + t * (col1[2] - col0[2]),
                        ];

                        edge_v[e as usize] = local_verts.len() as u32;
                        // residue_id is assigned in the post-process pass below.
                        local_verts.push(RibbonVertex { position, normal, color, residue_id: 0 });
                    }

                    // Emit triangles from the lookup table.
                    // Guard: skip triangles that reference edges not in edge_mask
                    // (guards against TRI_TABLE / EDGE_TABLE inconsistencies which
                    // would produce degenerate "line" triangles via the zeroed edge_v).
                    let tris = &TRI_TABLE[cube_idx as usize];
                    let mut ti = 0;
                    while ti < 15 && tris[ti] != -1 {
                        let e0 = tris[ti  ] as u32;
                        let e1 = tris[ti+1] as u32;
                        let e2 = tris[ti+2] as u32;
                        if (edge_mask & (1 << e0)) != 0
                            && (edge_mask & (1 << e1)) != 0
                            && (edge_mask & (1 << e2)) != 0
                        {
                            local_idxs.push(edge_v[e0 as usize]);
                            local_idxs.push(edge_v[e1 as usize]);
                            local_idxs.push(edge_v[e2 as usize]);
                        }
                        ti += 3;
                    }
                }
            }

            (local_verts, local_idxs)
        })
        .collect();

    // Merge per-slice results, offsetting indices by the running vertex count.
    let vert_start = vertices.len();
    let idx_start  = indices.len();
    for (local_verts, local_idxs) in slices {
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&local_verts);
        indices.extend(local_idxs.into_iter().map(|idx| idx + base));
    }

    // ── 6. Keep only the largest connected component ───────────────────────────
    // Marching Cubes generates per-slice vertex buffers with NO shared vertices
    // across slice boundaries.  Two adjacent cubes sharing a geometric edge
    // produce SEPARATE vertices at the same position, so the raw mesh is
    // topologically disconnected even though it looks continuous.
    //
    // Fix: first WELD duplicate vertices (same position → same index), then run
    // BFS on the welded mesh to find the true connected components.
    {
        let n_raw = vertices.len() - vert_start;
        if n_raw > 0 {
            // ── Step 1: Weld vertices ─────────────────────────────────────────
            // Quantise each position to a fixed grid (0.5 mm ≈ sub-Ångström)
            // and merge vertices that map to the same grid cell.
            const WELD_SCALE: f32 = 2048.0;
            let mut pos_map: HashMap<[i32; 3], u32> = HashMap::new();
            let mut welded_verts: Vec<RibbonVertex> = Vec::new();
            let mut raw_to_welded = vec![0u32; n_raw];
            for i in 0..n_raw {
                let v = &vertices[vert_start + i];
                let key = [
                    (v.position[0] * WELD_SCALE).round() as i32,
                    (v.position[1] * WELD_SCALE).round() as i32,
                    (v.position[2] * WELD_SCALE).round() as i32,
                ];
                let wi = *pos_map.entry(key).or_insert_with(|| {
                    let j = welded_verts.len() as u32;
                    welded_verts.push(*v);
                    j
                });
                raw_to_welded[i] = wi;
            }

            // Remap index buffer to welded vertex indices.
            let welded_idxs: Vec<u32> = indices[idx_start..]
                .iter()
                .map(|&i| raw_to_welded[(i - vert_start as u32) as usize])
                .collect();
            let n_welded = welded_verts.len();

            // ── Step 2: BFS on welded mesh ────────────────────────────────────
            let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n_welded];
            for tri in welded_idxs.chunks(3) {
                let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                adj[a].push(b as u32); adj[a].push(c as u32);
                adj[b].push(a as u32); adj[b].push(c as u32);
                adj[c].push(a as u32); adj[c].push(b as u32);
            }

            let mut comp = vec![u32::MAX; n_welded];
            let mut comp_sizes: Vec<usize> = Vec::new();
            let mut comp_id = 0u32;
            for start in 0..n_welded {
                if comp[start] != u32::MAX { continue; }
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(start);
                comp[start] = comp_id;
                let mut size = 0usize;
                while let Some(v) = queue.pop_front() {
                    size += 1;
                    for &nb in &adj[v] {
                        let nb = nb as usize;
                        if comp[nb] == u32::MAX {
                            comp[nb] = comp_id;
                            queue.push_back(nb);
                        }
                    }
                }
                comp_sizes.push(size);
                comp_id += 1;
            }

            // ── Step 3: Keep largest component ───────────────────────────────
            let largest = comp_sizes
                .iter()
                .enumerate()
                .max_by_key(|(_, &s)| s)
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            let removed: usize = comp_sizes.iter().enumerate()
                .filter(|&(i, _)| i as u32 != largest)
                .map(|(_, &s)| s)
                .sum();
            if removed > 0 {
                log::info!(
                    "surface: removed {} welded-vertices in {} small components",
                    removed,
                    comp_sizes.len() - 1
                );
            }

            // Filter index buffer: keep triangles whose welded vertices are all in `largest`.
            let kept_idxs: Vec<u32> = welded_idxs
                .chunks(3)
                .filter(|tri| comp[tri[0] as usize] == largest)
                .flat_map(|tri| tri.iter().copied())
                .collect();

            // ── Step 4: Compact welded vertex array ───────────────────────────
            let mut used = vec![false; n_welded];
            for &i in &kept_idxs { used[i as usize] = true; }

            let mut welded_remap = vec![0u32; n_welded];
            let mut final_verts: Vec<RibbonVertex> = Vec::new();
            let mut next = 0u32;
            for (i, &u) in used.iter().enumerate() {
                if u {
                    welded_remap[i] = next;
                    final_verts.push(welded_verts[i]);
                    next += 1;
                }
            }
            // Local (0-based) triangle indices for smoothing.
            let local_idxs: Vec<u32> = kept_idxs
                .iter()
                .map(|&i| welded_remap[i as usize])
                .collect();

            // ── Step 4.5: Close Marching-Cubes gaps ──────────────────────────
            // The MC output occasionally leaves a few tiny holes (missing
            // triangles). On a closed molecular surface every such hole is a
            // defect, so fill them all before smoothing — otherwise the gaps
            // show through as speckles, especially over a light background.
            let local_idxs = fill_all_holes(&mut final_verts, local_idxs, HOLE_MAX_EDGES);

            // ── Step 5: Taubin smoothing (removes grid-quantization bumps) ────
            if smooth_iters > 0 {
                smooth_surface_mesh(&mut final_verts, &local_idxs, smooth_iters);
            }

            let final_idxs: Vec<u32> = local_idxs
                .iter()
                .map(|&i| i + vert_start as u32)
                .collect();

            // Write back
            vertices.truncate(vert_start);
            vertices.extend_from_slice(&final_verts);
            indices.truncate(idx_start);
            indices.extend_from_slice(&final_idxs);
        }
    }

    // ── 7. Post-process: assign residue_id to each surface vertex ─────────────
    // Build a spatial hash of atom positions (cell size ~3 Å) then query
    // the nearest atom for each vertex. O(A) build + O(V × ~27) query.
    const CELL: f32 = 3.0;
    let mut spatial_hash: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (idx, (pos, _, _, _)) in atoms_data.iter().enumerate() {
        let key = (
            (pos.x / CELL).floor() as i32,
            (pos.y / CELL).floor() as i32,
            (pos.z / CELL).floor() as i32,
        );
        spatial_hash.entry(key).or_default().push(idx);
    }

    vertices[vert_start..].par_iter_mut().for_each(|v| {
        let p = Vec3::from(v.position);
        let cx = (p.x / CELL).floor() as i32;
        let cy = (p.y / CELL).floor() as i32;
        let cz = (p.z / CELL).floor() as i32;

        let mut best_dist2 = f32::MAX;
        let mut best_resid = 0u32;

        for ddx in -1..=1i32 {
            for ddy in -1..=1i32 {
                for ddz in -1..=1i32 {
                    if let Some(idxs) = spatial_hash.get(&(cx + ddx, cy + ddy, cz + ddz)) {
                        for &aidx in idxs {
                            let d2 = (atoms_data[aidx].0 - p).length_squared();
                            if d2 < best_dist2 {
                                best_dist2 = d2;
                                best_resid = atoms_data[aidx].2;
                            }
                        }
                    }
                }
            }
        }
        v.residue_id = best_resid;
    });
}

/// Taubin (λ/μ) smoothing of a welded surface mesh, followed by normal
/// recomputation. Positions are relaxed toward the neighbour average with an
/// alternating shrink (λ > 0) / un-shrink (μ < 0) step so the mesh keeps its
/// volume while grid-quantization bumps are removed.
///
/// `indices` are triangle indices in the local `verts` index space. The
/// recomputed geometric normal is re-oriented to agree with the original
/// gradient-based normal, so it stays correct regardless of triangle winding.
/// Fill every boundary-loop hole in a mesh by centroid-capping.
///
/// A watertight surface has no boundary edges: every undirected edge is shared
/// by two oppositely-wound triangles. Marching Cubes occasionally drops a
/// triangle, leaving a small open loop that renders as a see-through hole. This
/// walks the boundary edges into loops (consuming every outgoing edge per vertex
/// so non-manifold junctions are handled) and caps each with a new centroid
/// vertex fanned to reverse-wound sealing triangles. A centroid cap closes
/// non-planar and self-touching loops that a single-anchor fan would leave open.
/// Unlike the pocket-mode filler it fills *all* loops (a closed surface has no
/// intended rim); loops longer than `max_edges` are left alone to avoid sheets.
fn fill_all_holes(verts: &mut Vec<RibbonVertex>, mut idxs: Vec<u32>, max_edges: usize) -> Vec<u32> {
    use std::collections::{HashMap, HashSet};
    if idxs.len() < 3 {
        return idxs;
    }

    // All directed edges present in the mesh.
    let mut dir_edges: HashSet<(u32, u32)> = HashSet::with_capacity(idxs.len());
    for t in idxs.chunks(3) {
        dir_edges.insert((t[0], t[1]));
        dir_edges.insert((t[1], t[2]));
        dir_edges.insert((t[2], t[0]));
    }

    // A directed edge (u,v) is a boundary edge when its reverse is absent. A
    // vertex may have several outgoing boundary edges (non-manifold junctions),
    // so store them all and consume each exactly once when walking loops.
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut boundary_starts: Vec<u32> = Vec::new();
    for &(u, v) in &dir_edges {
        if !dir_edges.contains(&(v, u)) {
            adj.entry(u).or_default().push(v);
            boundary_starts.push(u);
        }
    }
    if adj.is_empty() {
        return idxs;
    }
    let n_boundary: usize = adj.values().map(|a| a.len()).sum();

    // Per-vertex pointer into its unused outgoing boundary edges.
    let mut ptr: HashMap<u32, usize> = HashMap::new();
    let mut n_filled = 0usize;
    let mut n_skipped = 0usize;
    for start in boundary_starts {
        // Walk loops out of `start` until all its edges are consumed.
        loop {
            let p0 = *ptr.get(&start).unwrap_or(&0);
            if p0 >= adj.get(&start).map_or(0, |a| a.len()) {
                break;
            }
            let mut lp = Vec::new();
            let mut cur = start;
            let mut closed = false;
            loop {
                let p = ptr.entry(cur).or_insert(0);
                let outs = match adj.get(&cur) {
                    Some(o) if *p < o.len() => o,
                    _ => break, // dead end: an open path (non-manifold junction)
                };
                let nxt = outs[*p];
                *p += 1;
                lp.push(cur);
                cur = nxt;
                if cur == start {
                    closed = true;
                    break; // loop closed
                }
            }
            // Cap the loop with a centroid vertex, sealing each boundary edge
            // lp[i]→lp[i+1] with the reverse-wound triangle (lp[i+1], lp[i], c).
            // For a closed loop the last edge wraps lp[last]→lp[0]; an open path
            // (rare, from non-manifold junctions) is capped without the wrap.
            if lp.len() >= 3 && lp.len() <= max_edges {
                let mut pos = Vec3::ZERO;
                let mut nrm = Vec3::ZERO;
                let mut col = [0.0f32; 3];
                for &vi in &lp {
                    let v = &verts[vi as usize];
                    pos += Vec3::from(v.position);
                    nrm += Vec3::from(v.normal);
                    col[0] += v.color[0];
                    col[1] += v.color[1];
                    col[2] += v.color[2];
                }
                let inv = 1.0 / lp.len() as f32;
                let c = verts.len() as u32;
                verts.push(RibbonVertex {
                    position: (pos * inv).to_array(),
                    normal: nrm.normalize_or_zero().to_array(),
                    color: [col[0] * inv, col[1] * inv, col[2] * inv],
                    residue_id: verts[lp[0] as usize].residue_id,
                });
                let edges = if closed { lp.len() } else { lp.len() - 1 };
                for i in 0..edges {
                    let a = lp[i];
                    let b = lp[(i + 1) % lp.len()];
                    idxs.push(b);
                    idxs.push(a);
                    idxs.push(c);
                }
                n_filled += 1;
            } else if lp.len() >= 3 {
                n_skipped += 1;
            }
        }
    }

    log::debug!(
        "fill_all_holes: {} boundary edges, filled {} loops, skipped {} (too large)",
        n_boundary, n_filled, n_skipped,
    );
    idxs
}

fn smooth_surface_mesh(verts: &mut [RibbonVertex], indices: &[u32], iterations: usize) {
    let n = verts.len();
    if n == 0 || indices.len() < 3 {
        return;
    }

    // ── Vertex adjacency (unique undirected edges) ──────────────────────────
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for tri in indices.chunks(3) {
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            if seen.insert(key) {
                adj[a as usize].push(b);
                adj[b as usize].push(a);
            }
        }
    }

    let mut pos: Vec<Vec3> = verts.iter().map(|v| Vec3::from(v.position)).collect();
    let mut tmp = pos.clone();

    // One Laplacian pass with the given relaxation factor: src → dst.
    let pass = |factor: f32, src: &[Vec3], dst: &mut [Vec3], adj: &[Vec<u32>]| {
        dst.par_iter_mut().enumerate().for_each(|(i, out)| {
            let nbrs = &adj[i];
            if nbrs.is_empty() {
                *out = src[i];
                return;
            }
            let mut avg = Vec3::ZERO;
            for &nb in nbrs {
                avg += src[nb as usize];
            }
            avg /= nbrs.len() as f32;
            *out = src[i] + factor * (avg - src[i]);
        });
    };

    for _ in 0..iterations {
        pass(SMOOTH_LAMBDA, &pos, &mut tmp, &adj);
        pass(SMOOTH_MU, &tmp, &mut pos, &adj);
    }

    // ── Recompute normals (area-weighted face normals) ──────────────────────
    // Marching Cubes triangle winding is not globally consistent, so a raw
    // area-weighted sum lets oppositely-wound neighbouring faces cancel and
    // leaves some vertices with a garbage-direction normal — which the surface
    // shader then renders as a dark back-face speckle ("holes"). Orient each
    // face normal to agree with its vertices' original gradient normals before
    // accumulating, so the sum is winding-independent.
    let mut normals = vec![Vec3::ZERO; n];
    for tri in indices.chunks(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let mut fnrm = (pos[b] - pos[a]).cross(pos[c] - pos[a]);
        let ref_n = Vec3::from(verts[a].normal)
            + Vec3::from(verts[b].normal)
            + Vec3::from(verts[c].normal);
        if fnrm.dot(ref_n) < 0.0 {
            fnrm = -fnrm;
        }
        normals[a] += fnrm;
        normals[b] += fnrm;
        normals[c] += fnrm;
    }

    for i in 0..n {
        verts[i].position = pos[i].to_array();
        let len = normals[i].length();
        if len > 1e-8 {
            // The accumulated normal is already oriented outward (each face was
            // aligned to its vertices' gradient normals above). Do NOT re-flip
            // against a single vertex's gradient normal here: that reference is
            // occasionally noisy/inward and would flip an otherwise-correct
            // normal inward, producing a dark back-face speckle.
            verts[i].normal = (normals[i] / len).to_array();
        }
    }
}

