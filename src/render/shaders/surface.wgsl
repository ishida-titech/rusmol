// Surface shader — PBR (GGX Cook-Torrance), single-layer alpha-blend.
// Renders into the MSAA HDR target. ACES tone mapping is done in post.wgsl.

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
    @location(0) position:   vec3<f32>,
    @location(1) normal:     vec3<f32>,
    @location(2) color:      vec3<f32>,
    @location(3) residue_id: u32,
}

struct VertOut {
    @builtin(position)              clip_pos:   vec4<f32>,
    @location(0)                    world_pos:  vec3<f32>,
    @location(1)                    world_nrm:  vec3<f32>,
    @location(2)                    color:      vec3<f32>,
    @location(3) @interpolate(flat) residue_id: u32,
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

// Surface uses a 0.65× IBL scale so the semi-transparent surface is subtler than solid geometry.
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
    return (diffuse_ibl + specular_ibl) * u.ibl_intensity * 0.65;
}

// ── Vertex ────────────────────────────────────────────────────────────────────

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    out.clip_pos   = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos  = in.position;
    out.world_nrm  = in.normal;
    out.color      = in.color;
    out.residue_id = in.residue_id;
    return out;
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let V = normalize(u.camera_pos - in.world_pos);
    let raw_N = normalize(in.world_nrm);
    // MC gradient normals always point outward. When the outward normal faces
    // away from the camera we are looking at the *inside* (back face) of the
    // mesh, e.g. through an opening in the surface.
    let is_back = dot(raw_N, V) < 0.0;
    // Camera-facing normal for shading.
    var N = select(-raw_N, raw_N, dot(raw_N, V) >= 0.0);

    var albedo = in.color;
    var rough  = u.roughness;

    // Back faces get a dark matte gray with a procedural bump so the inside of
    // the surface is obviously different from the (smooth, colored) front.
    if is_back {
        albedo = vec3<f32>(0.16, 0.16, 0.18);
        rough  = 0.9;
        let freq   = 2.5;
        let grad   = vec3<f32>(cos(in.world_pos.x * freq),
                               cos(in.world_pos.y * freq),
                               cos(in.world_pos.z * freq));
        let grad_t = grad - N * dot(grad, N);   // tangential component
        N = normalize(N - grad_t * 0.6);        // perturb normal → bumpy look
    }

    let L = normalize(u.light_dir);

    let shadow = mix(1.0, shadow_factor(in.world_pos), u.shadow_strength);
    var color = shadow * pbr_direct(N, V, L, albedo, rough, u.metallic, u.light_intensity)
             + pbr_ibl(N, V, albedo, rough, u.metallic);

    // Light 2
    if u.light2_intensity > 0.0 {
        let L2 = normalize(vec3<f32>(u.light2_dir_xy, u.light2_dir_z));
        color += pbr_direct(N, V, L2, albedo, rough, u.metallic, u.light2_intensity);
    }

    // Fresnel edge darkening (cartoon silhouette)
    let nv        = max(dot(N, V), 0.0);
    let edge_dark = clamp(pow(1.0 - nv, 6.0) * 0.55 * u.edge_strength, 0.0, 1.0);
    color *= 1.0 - edge_dark;

    // Residue highlight: orange rim glow (front faces only)
    if !is_back && u.picked_residue_id != 0u && in.residue_id == u.picked_residue_id {
        let rim = pow(1.0 - nv, 3.0);
        color += rim * vec3<f32>(1.0, 0.6, 0.0) * 1.5;
    }

    let alpha = clamp(u.surface_alpha, 0.0, 1.0);
    return vec4<f32>(color, alpha);
}
