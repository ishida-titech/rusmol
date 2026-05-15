// Sphere Impostor shader — PBR (GGX Cook-Torrance).
// Billboard quads (6 verts/instance, no mesh buffer) with ray-sphere intersection
// in the fragment shader for correct normals, depth, and physically based lighting.

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
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var shadow_map:  texture_depth_2d;
@group(1) @binding(1) var shadow_samp: sampler_comparison;

struct InstIn {
    @location(0) inst_pos:    vec3<f32>,
    @location(1) inst_radius: f32,
    @location(2) inst_color:  vec3<f32>,
    @location(3) edge_boost:  f32,
}

struct VertOut {
    @builtin(position) clip_pos:      vec4<f32>,
    @location(0)       sphere_center: vec3<f32>,
    @location(1)       sphere_radius: f32,
    @location(2)       color:         vec3<f32>,
    @location(3)       billboard_pos: vec3<f32>,
    @location(4)       edge_boost:    f32,
}

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0)         color: vec4<f32>,
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

// Cook-Torrance BRDF with half-Lambert diffuse wrap (direct light only, no ambient).
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

// Hemispherical IBL: sky_color from above (camera_up), ground_color from below.
// Diffuse irradiance + approximate specular reflection from the environment.
fn pbr_ibl(N: vec3<f32>, V: vec3<f32>, albedo: vec3<f32>, roughness: f32, metallic: f32) -> vec3<f32> {
    let F0 = mix(vec3(0.04), albedo, metallic);

    // Diffuse IBL: hemisphere weighted by normal vs camera_up
    let NdotUp     = dot(N, u.camera_up);
    let sky_w      = clamp(0.5 + 0.5 * NdotUp, 0.0, 1.0);
    let irradiance = mix(u.ground_color, u.sky_color, sky_w);
    let F_diff     = F_Schlick(max(dot(N, V), 0.0), F0);
    let kD         = (vec3(1.0) - F_diff) * (1.0 - metallic);
    let diffuse_ibl = kD * albedo * irradiance;

    // Specular IBL: reflection direction samples hemisphere
    let R          = reflect(-V, N);
    let RdotUp     = clamp(0.5 + 0.5 * dot(R, u.camera_up), 0.0, 1.0);
    let env_col    = mix(u.ground_color, u.sky_color, RdotUp);
    let spec_fac   = (1.0 - roughness) * (1.0 - roughness);
    let specular_ibl = F_diff * env_col * spec_fac;

    return (diffuse_ibl + specular_ibl) * u.ibl_intensity;
}

// ── Vertex ────────────────────────────────────────────────────────────────────

@vertex
fn vs_main(inst: InstIn, @builtin(vertex_index) vid: u32) -> VertOut {
    let offsets = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
        vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    );
    let uv = offsets[vid];

    let half_ext = inst.inst_radius * 1.15;
    let corner   = inst.inst_pos
        + u.camera_right * (uv.x * half_ext)
        + u.camera_up    * (uv.y * half_ext);

    var out: VertOut;
    out.clip_pos      = u.view_proj * vec4<f32>(corner, 1.0);
    out.sphere_center = inst.inst_pos;
    out.sphere_radius = inst.inst_radius;
    out.color         = inst.inst_color;
    out.billboard_pos = corner;
    out.edge_boost    = inst.edge_boost;
    return out;
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertOut) -> FragOut {
    // Ray-sphere intersection
    let ray_orig = u.camera_pos;
    let ray_dir  = normalize(in.billboard_pos - u.camera_pos);

    let oc   = ray_orig - in.sphere_center;
    let b    = dot(oc, ray_dir);
    let c    = dot(oc, oc) - in.sphere_radius * in.sphere_radius;
    let disc = b * b - c;
    if disc < 0.0 { discard; }

    let sq     = sqrt(disc);
    let t_near = -b - sq;
    let t_far  = -b + sq;
    if t_far < 0.0 { discard; }

    let t   = select(t_far, t_near, t_near >= 0.0);
    let hit = ray_orig + t * ray_dir;
    let N   = (hit - in.sphere_center) / in.sphere_radius;

    let L = normalize(u.light_dir);
    let V = normalize(u.camera_pos - hit);

    let shadow = mix(1.0, shadow_factor(hit), u.shadow_strength);
    var color = shadow * pbr_direct(N, V, L, in.color, u.roughness, u.metallic, u.light_intensity)
             + pbr_ibl(N, V, in.color, u.roughness, u.metallic);

    // Fresnel edge darkening (stylistic silhouette)
    let nv        = max(dot(N, V), 0.0);
    let eff_edge  = max(u.edge_strength, in.edge_boost);
    let edge_dark = clamp(pow(1.0 - nv, 6.0) * 0.40 * eff_edge, 0.0, 1.0);
    color *= 1.0 - edge_dark;

    // Correct clip-space depth from sphere surface
    let clip_hit = u.view_proj * vec4<f32>(hit, 1.0);

    var out: FragOut;
    out.depth = clip_hit.z / clip_hit.w;
    out.color = vec4<f32>(color, 1.0);
    return out;
}
