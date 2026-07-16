// Cylinder shader — PBR (GGX Cook-Torrance).

struct Uniforms {
    view_proj:         mat4x4<f32>,   // offset   0
    light_dir:         vec3<f32>,     // offset  64
    picked_residue_id: u32,           // offset  76
    camera_pos:        vec3<f32>,     // offset  80
    light_intensity:   f32,           // offset  92
    inv_proj:          mat4x4<f32>,   // offset  96
    screen_size:       vec2<f32>,     // offset 160
    surface_alpha:     f32,           // offset 168
    edge_strength:     f32,           // offset 172
    bg_color:          vec4<f32>,     // offset 176
    camera_right:      vec3<f32>,     // offset 192
    roughness:         f32,           // offset 204
    camera_up:         vec3<f32>,     // offset 208
    metallic:          f32,           // offset 220
    sky_color:         vec3<f32>,     // offset 224
    ibl_intensity:     f32,           // offset 236
    ground_color:      vec3<f32>,     // offset 240
    shadow_strength:   f32,           // offset 252
    light_view_proj:   mat4x4<f32>,  // offset 256
    bloom_threshold:   f32,           // offset 320
    bloom_intensity:   f32,           // offset 324
    light2_dir_xy:     vec2<f32>,     // offset 328
    light2_dir_z:      f32,           // offset 336
    light2_intensity:  f32,           // offset 340
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var shadow_map:  texture_depth_2d;
@group(1) @binding(1) var shadow_samp: sampler_comparison;

struct VertIn {
    @location(0) position:  vec3<f32>,
    @location(1) normal:    vec3<f32>,
    // instance
    @location(2) start_r:   vec4<f32>,  // xyz=start, w=radius
    @location(3) end_pad:   vec4<f32>,  // xyz=end,   w=unused
    @location(4) color_pad: vec4<f32>,  // xyz=color, w=unused
}

struct VertOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec3<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       color:        vec3<f32>,
    @location(3)       edge_boost:   f32,
}

/// Rodrigues rotation: rotate `v` so that Y-axis maps to `dir`.
fn rotate_y_to(v: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let y = vec3<f32>(0.0, 1.0, 0.0);
    let c = dot(y, dir);
    if c > 0.9999 { return v; }
    if c < -0.9999 { return vec3<f32>(v.x, -v.y, v.z); }

    let ax  = normalize(cross(y, dir));
    let s   = sqrt(max(0.0, 1.0 - c * c));
    let t   = 1.0 - c;
    let ax_ = ax.x; let ay_ = ax.y; let az_ = ax.z;

    let rot = mat3x3<f32>(
        vec3<f32>(t*ax_*ax_ + c,       t*ax_*ay_ + s*az_, t*ax_*az_ - s*ay_),
        vec3<f32>(t*ax_*ay_ - s*az_,   t*ay_*ay_ + c,     t*ay_*az_ + s*ax_),
        vec3<f32>(t*ax_*az_ + s*ay_,   t*ay_*az_ - s*ax_, t*az_*az_ + c    )
    );
    return rot * v;
}

// ── Shadow sampling ──────────────────────────────────────────────────────────

fn shadow_factor(world_pos: vec3<f32>) -> f32 {
    let lc  = u.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lc.xyz / lc.w;
    let uv  = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 { return 1.0; }
    let depth = ndc.z - 0.002;
    let tx = 1.0 / 2048.0;
    var s = 0.0;
    for (var x = -1i; x <= 1i; x++) {
        for (var y = -1i; y <= 1i; y++) {
            s += textureSampleCompare(shadow_map, shadow_samp,
                uv + vec2<f32>(f32(x), f32(y)) * tx, depth);
        }
    }
    return s / 9.0;
}

// ── PBR helpers ───────────────────────────────────────────────────────────────

const PI: f32 = 3.14159265358979;

fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn G_SchlickGGX(NdotX: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotX / (NdotX * (1.0 - k) + k);
}

fn G_Smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

fn F_Schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn pbr_direct(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
    albedo: vec3<f32>, roughness: f32, metallic: f32,
    light_intensity: f32,
) -> vec3<f32> {
    let H     = normalize(V + L);
    let NdotH = max(dot(N, H), 0.0);
    let NdotV = max(dot(N, V), 0.001);
    let NdotL = max(dot(N, L), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    let F0 = mix(vec3(0.04), albedo, metallic);

    let D = D_GGX(NdotH, roughness);
    let G = G_Smith(NdotV, NdotL, roughness);
    let F = F_Schlick(HdotV, F0);

    let spec    = (D * G * F) / max(4.0 * NdotV * NdotL, 0.0001);
    let kD      = (vec3(1.0) - F) * (1.0 - metallic);
    let wrap    = NdotL * 0.55 + 0.45;
    let diffuse = kD * albedo * wrap;

    return (diffuse + spec * NdotL) * light_intensity;
}

fn pbr_ibl(N: vec3<f32>, V: vec3<f32>, albedo: vec3<f32>, roughness: f32, metallic: f32) -> vec3<f32> {
    let F0         = mix(vec3(0.04), albedo, metallic);
    let NdotUp     = dot(N, u.camera_up);
    let sky_w      = clamp(0.5 + 0.5 * NdotUp, 0.0, 1.0);
    let irradiance = mix(u.ground_color, u.sky_color, sky_w);
    let F_diff     = F_Schlick(max(dot(N, V), 0.0), F0);
    let kD         = (vec3(1.0) - F_diff) * (1.0 - metallic);
    let diffuse_ibl  = kD * albedo * irradiance;
    let R            = reflect(-V, N);
    let RdotUp       = clamp(0.5 + 0.5 * dot(R, u.camera_up), 0.0, 1.0);
    let env_col      = mix(u.ground_color, u.sky_color, RdotUp);
    let spec_fac     = (1.0 - roughness) * (1.0 - roughness);
    let specular_ibl = F_diff * env_col * spec_fac;
    return (diffuse_ibl + specular_ibl) * u.ibl_intensity;
}

// ── Vertex ────────────────────────────────────────────────────────────────────

@vertex
fn vs_main(in: VertIn) -> VertOut {
    let start  = in.start_r.xyz;
    let radius = in.start_r.w;
    let end    = in.end_pad.xyz;
    let color  = in.color_pad.xyz;

    let axis   = end - start;
    let length = length(axis);
    let dir    = select(vec3<f32>(0.0, 1.0, 0.0), normalize(axis), length > 0.0001);
    let center = (start + end) * 0.5;

    let scaled = vec3<f32>(
        in.position.x * radius,
        in.position.y * length,
        in.position.z * radius,
    );
    let world_pos    = rotate_y_to(scaled, dir) + center;
    let mesh_normal  = normalize(vec3<f32>(in.normal.x, 0.0, in.normal.z));
    let world_normal = rotate_y_to(mesh_normal, dir);

    var out: VertOut;
    out.clip_pos     = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos    = world_pos;
    out.world_normal = world_normal;
    out.color        = color;
    out.edge_boost   = in.color_pad.w;
    return out;
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let L = normalize(u.light_dir);
    let V = normalize(u.camera_pos - in.world_pos);

    // Ligand overlay geometry (edge_boost = 1) is excluded from shadows and (via
    // the alpha marker below) from SSAO, so a translucent protein surface only
    // dims the ligand through its own transparency — no extra shadow/AO murk.
    let is_ligand = in.edge_boost >= 0.5;
    let recv_shadow = select(u.shadow_strength, 0.0, is_ligand);
    let shadow = mix(1.0, shadow_factor(in.world_pos), recv_shadow);
    var color = shadow * pbr_direct(N, V, L, in.color, u.roughness, u.metallic, u.light_intensity)
             + pbr_ibl(N, V, in.color, u.roughness, u.metallic);

    // Light 2
    if u.light2_intensity > 0.0 {
        let L2 = normalize(vec3<f32>(u.light2_dir_xy, u.light2_dir_z));
        color += pbr_direct(N, V, L2, in.color, u.roughness, u.metallic, u.light2_intensity);
    }

    // Fresnel edge darkening (stylistic silhouette). Ligand bonds (edge_boost=1)
    // get a wider, darker rim so they stand out from the protein.
    let nv        = max(dot(N, V), 0.0);
    let eff_edge  = max(u.edge_strength, in.edge_boost);
    let edge_pow  = mix(6.0, 1.0, in.edge_boost);
    let edge_amt  = mix(0.40, 1.0, in.edge_boost);
    let edge_dark = clamp(pow(1.0 - nv, edge_pow) * edge_amt * eff_edge, 0.0, 1.0);
    color *= 1.0 - edge_dark;

    // Alpha is a post-pass mask: 1 = normal geometry, 0.25 = ligand (skip SSAO).
    // (Not 0 — the post pass reserves alpha 0 for the empty background.)
    return vec4<f32>(color, select(1.0, 0.25, is_ligand));
}
