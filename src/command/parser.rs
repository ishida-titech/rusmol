use std::path::PathBuf;

use super::selection::parse as parse_sel;
use super::{ColorSpec, Command};
use crate::scene::object::RepresentationType;

pub fn parse_command(input: &str) -> Result<Command, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("empty command".into());
    }

    let (cmd, rest) = split_first_word(input);
    let rest = rest.trim();

    match cmd.to_lowercase().as_str() {
        "load"             => parse_load(rest),
        "select" | "sel"   => parse_select(rest),
        "show"             => parse_show_hide(rest, true),
        "hide"             => parse_show_hide(rest, false),
        "color" | "colour" => parse_color(rest),
        "enable"           => Ok(Command::Enable(rest.to_string())),
        "disable"          => Ok(Command::Disable(rest.to_string())),
        "delete" | "del"   => Ok(Command::Delete(rest.to_string())),
        "zoom" | "z"       => {
            let sel = if rest.is_empty() { None } else { Some(rest.to_string()) };
            Ok(Command::Zoom { sel })
        }
        "reset"            => Ok(Command::Reset),
        "bg" | "background" | "bgcolor" => {
            if rest.is_empty() {
                return Err("bg: expected color name".into());
            }
            match named_color(rest) {
                Some(rgb) => Ok(Command::Background(rgb)),
                None      => Err(format!("bg: unknown color '{rest}'")),
            }
        }
        "light" => parse_light(rest),
        "light2" => parse_light2(rest),
        "set"   => parse_set(rest),
        "get"   => {
            let name = rest.trim();
            Ok(Command::Get { name: if name.is_empty() { None } else { Some(name.to_string()) } })
        }
        "png" | "screenshot" => {
            if rest.is_empty() {
                return Err("png: expected a file path".into());
            }
            Ok(Command::Png { path: PathBuf::from(rest) })
        }
        "docktrace" => parse_docktrace(rest),
        "help" | "h" | "?" => Ok(Command::Help),
        "quit" | "q" | "exit" => Ok(Command::Quit),
        other => Err(format!("unknown command: '{other}'")),
    }
}

// ── Sub-parsers ───────────────────────────────────────────────────────────────

fn parse_load(rest: &str) -> Result<Command, String> {
    if rest.is_empty() {
        return Err("load: expected a file path".into());
    }
    let parts = split_comma(rest, 2);
    let path = PathBuf::from(parts[0].trim());
    let name = parts.get(1).map(|s| s.trim().to_string());
    Ok(Command::Load { path, name })
}

fn parse_select(rest: &str) -> Result<Command, String> {
    // Format: [name,] expr   — if no comma, name = "sele"
    let parts = split_comma(rest, 2);
    let (name, expr_str) = if parts.len() == 2 {
        (parts[0].trim().to_string(), parts[1].trim())
    } else {
        ("sele".to_string(), rest.trim())
    };
    let expr = parse_sel(expr_str).map_err(|e| format!("select: {e}"))?;
    Ok(Command::Select { name, expr })
}

fn parse_show_hide(rest: &str, show: bool) -> Result<Command, String> {
    if rest.is_empty() {
        return Err(format!("{}: expected representation name", if show { "show" } else { "hide" }));
    }
    let parts = split_comma(rest, 2);
    let repr_str = parts[0].trim();
    let sel = parts.get(1).map(|s| s.trim().to_string());

    // "everything" with no sel = apply to all
    if repr_str.eq_ignore_ascii_case("everything") {
        // show/hide all representations — handled in executor
        let repr = RepresentationType::BallAndStick;
        if show {
            return Ok(Command::Show { repr, sel });
        } else {
            return Ok(Command::Hide { repr, sel });
        }
    }

    let repr = parse_repr(repr_str)?;
    if show {
        Ok(Command::Show { repr, sel })
    } else {
        Ok(Command::Hide { repr, sel })
    }
}

fn parse_color(rest: &str) -> Result<Command, String> {
    if rest.is_empty() {
        return Err("color: expected color name".into());
    }
    let parts = split_comma(rest, 2);
    let color_str = parts[0].trim();
    let sel = parts.get(1).map(|s| s.trim().to_string());
    let spec = parse_color_spec(color_str)?;
    Ok(Command::Color { spec, sel })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_repr(s: &str) -> Result<RepresentationType, String> {
    match s.to_lowercase().as_str() {
        "ball_stick" | "ball-stick" | "bs" | "ball_and_stick" | "spheres" => {
            Ok(RepresentationType::BallAndStick)
        }
        "stick" | "sticks" => Ok(RepresentationType::Stick),
        "backbone" | "trace" | "ca_trace" | "ca" => Ok(RepresentationType::Backbone),
        "ribbon" | "cartoon" => Ok(RepresentationType::Ribbon),
        "surface"            => Ok(RepresentationType::Surface),
        "lines" | "line" | "wire" => Ok(RepresentationType::Lines),
        other => Err(format!("unknown representation: '{other}'")),
    }
}

fn parse_color_spec(s: &str) -> Result<ColorSpec, String> {
    match s.to_lowercase().as_str() {
        "element" | "cpk" => return Ok(ColorSpec::Element),
        "chain" | "chainbows" => return Ok(ColorSpec::Chain),
        "ss" | "secondary" | "secondary_structure" => return Ok(ColorSpec::SecondaryStructure),
        "spectrum" | "rainbow" => return Ok(ColorSpec::Spectrum),
        "b" | "b_factor" | "bfactor" | "tempfactor" => return Ok(ColorSpec::BFactor),
        _ => {}
    }
    if let Some(rgb) = named_color(s) {
        return Ok(ColorSpec::Rgb(rgb));
    }
    Err(format!("unknown color: '{s}'"))
}

/// Parse: `set name, value[, sel]`
/// Supported:
///   transparency / surface_transparency  → Set { name, value: f32 }
///   surface_color / cartoon_color        → SetColor { rep, color, sel }
fn parse_set(rest: &str) -> Result<Command, String> {
    if rest.is_empty() {
        return Err("set: expected name, value".into());
    }
    let parts = split_comma(rest, 3);
    if parts.len() < 2 {
        return Err("set: expected 'set name, value'".into());
    }
    let name = parts[0].trim().to_lowercase();
    let val_str = parts[1].trim();

    match name.as_str() {
        "surface_color" | "cartoon_color" | "ribbon_color" => {
            let sel = parts.get(2).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
            let rep = if name == "surface_color" { "surface" } else { "ribbon" }.to_string();
            let color = if val_str.eq_ignore_ascii_case("default") {
                None
            } else {
                Some(named_color(val_str)
                    .ok_or_else(|| format!("set: unknown color '{val_str}'"))?)
            };
            Ok(Command::SetColor { rep, color, sel })
        }
        "surface_type" => {
            // Pass as a Set command with a sentinel value; the actual string
            // is handled in app.rs. We encode: gaussian=0, ses=1.
            let value = match val_str.to_lowercase().as_str() {
                "gaussian" | "gauss" => 0.0,
                "ses" | "connolly" | "molecular" => 1.0,
                other => return Err(format!("set: unknown surface_type '{other}' (use gaussian or ses)")),
            };
            Ok(Command::Set { name, value })
        }
        "transparency" | "surface_transparency" | "edge_strength" | "roughness" | "metallic"
        | "ibl_intensity" | "shadow_strength" | "shadow"
        | "bloom_threshold" | "bloom_intensity" | "bloom"
        | "surface_quality" | "surface_smooth"
        | "light_intensity" | "light_elevation" | "light_azimuth"
        | "light2_intensity" | "light2_elevation" | "light2_azimuth" => {
            let value: f32 = val_str
                .parse()
                .map_err(|_| format!("set: '{val_str}' is not a number"))?;
            Ok(Command::Set { name, value })
        }
        other => Err(format!("set: unknown setting '{other}'")),
    }
}

/// Parse: `light [intensity <f>] [elevation <f>] [azimuth <f>]`
/// All three parameters are optional and can be combined in any order.
fn parse_light(rest: &str) -> Result<Command, String> {
    let mut intensity: Option<f32> = None;
    let mut elevation: Option<f32> = None;
    let mut azimuth:   Option<f32> = None;

    let mut tokens = rest.split_whitespace();
    while let Some(key) = tokens.next() {
        let val_str = tokens.next()
            .ok_or_else(|| format!("light: expected value after '{key}'"))?;
        let val: f32 = val_str.parse()
            .map_err(|_| format!("light: '{val_str}' is not a number"))?;
        match key.to_lowercase().as_str() {
            "intensity" | "i" => intensity = Some(val.max(0.0)),
            "elevation" | "el" => elevation = Some(val.clamp(-90.0, 90.0)),
            "azimuth"   | "az" => azimuth   = Some(val),
            other => return Err(format!("light: unknown parameter '{other}'")),
        }
    }

    if intensity.is_none() && elevation.is_none() && azimuth.is_none() {
        return Err("light: expected intensity, elevation, or azimuth".into());
    }
    Ok(Command::Light { intensity, elevation, azimuth })
}

/// Parse: `light2 [intensity <f>] [elevation <f>] [azimuth <f>]`
fn parse_light2(rest: &str) -> Result<Command, String> {
    let mut intensity: Option<f32> = None;
    let mut elevation: Option<f32> = None;
    let mut azimuth:   Option<f32> = None;

    let mut tokens = rest.split_whitespace();
    while let Some(key) = tokens.next() {
        let val_str = tokens.next()
            .ok_or_else(|| format!("light2: expected value after '{key}'"))?;
        let val: f32 = val_str.parse()
            .map_err(|_| format!("light2: '{val_str}' is not a number"))?;
        match key.to_lowercase().as_str() {
            "intensity" | "i" => intensity = Some(val.max(0.0)),
            "elevation" | "el" => elevation = Some(val.clamp(-90.0, 90.0)),
            "azimuth"   | "az" => azimuth   = Some(val),
            other => return Err(format!("light2: unknown parameter '{other}'")),
        }
    }

    if intensity.is_none() && elevation.is_none() && azimuth.is_none() {
        return Err("light2: expected intensity, elevation, or azimuth".into());
    }
    Ok(Command::Light2 { intensity, elevation, azimuth })
}

fn named_color(s: &str) -> Option<[f32; 3]> {
    match s.to_lowercase().as_str() {
        "red"             => Some([1.00, 0.12, 0.12]),
        "green"           => Some([0.13, 0.70, 0.13]),
        "blue"            => Some([0.12, 0.47, 0.71]),
        "white"           => Some([1.00, 1.00, 1.00]),
        "black"           => Some([0.00, 0.00, 0.00]),
        "yellow"          => Some([1.00, 1.00, 0.00]),
        "orange"          => Some([1.00, 0.55, 0.00]),
        "purple" | "violet" => Some([0.55, 0.00, 0.55]),
        "magenta"         => Some([1.00, 0.00, 1.00]),
        "cyan"            => Some([0.00, 0.80, 0.80]),
        "grey" | "gray"   => Some([0.50, 0.50, 0.50]),
        "pink"            => Some([1.00, 0.75, 0.80]),
        "salmon"          => Some([0.98, 0.50, 0.45]),
        "wheat"           => Some([0.96, 0.87, 0.70]),
        "teal"            => Some([0.00, 0.50, 0.50]),
        "marine"          => Some([0.00, 0.45, 1.00]),
        "forest"          => Some([0.13, 0.55, 0.13]),
        "limon"           => Some([0.75, 1.00, 0.25]),
        _                 => None,
    }
}

/// Split `s` on the first `max_parts-1` commas, trimming each part.
fn split_comma(s: &str, max_parts: usize) -> Vec<&str> {
    let mut result = Vec::new();
    let mut remaining = s;
    for _ in 0..max_parts - 1 {
        if let Some(idx) = remaining.find(',') {
            result.push(&remaining[..idx]);
            remaining = &remaining[idx + 1..];
        } else {
            break;
        }
    }
    result.push(remaining);
    result
}

fn split_first_word(s: &str) -> (&str, &str) {
    if let Some(pos) = s.find(|c: char| c.is_whitespace()) {
        (&s[..pos], &s[pos..])
    } else {
        (s, "")
    }
}

fn parse_docktrace(rest: &str) -> Result<Command, String> {
    if rest.is_empty() {
        return Err("docktrace: expected <trace_file>, <ligand_pdbqt>".into());
    }
    let parts = split_comma(rest, 2);
    if parts.len() < 2 {
        return Err("docktrace: expected <trace_file>, <ligand_pdbqt>".into());
    }
    let trace_path = PathBuf::from(parts[0].trim());
    let ligand_path = PathBuf::from(parts[1].trim());
    Ok(Command::DockTrace { trace_path, ligand_path })
}

// ── Selection helper for executor use ─────────────────────────────────────────


// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Command;
    use crate::scene::object::RepresentationType;

    fn ok(s: &str) -> Command {
        parse_command(s).unwrap_or_else(|e| panic!("parse failed for '{s}': {e}"))
    }

    fn err(s: &str) {
        assert!(parse_command(s).is_err(), "expected error for: '{s}'");
    }

    // ── load ─────────────────────────────────────────────────────────────────

    #[test]
    fn load_path_only() {
        assert!(matches!(ok("load foo.pdb"), Command::Load { name: None, .. }));
    }

    #[test]
    fn load_with_name() {
        match ok("load foo.pdb, myobj") {
            Command::Load { name: Some(n), .. } => assert_eq!(n, "myobj"),
            _ => panic!("expected Load with name"),
        }
    }

    #[test]
    fn load_requires_path() {
        err("load");
    }

    // ── show / hide ───────────────────────────────────────────────────────────

    #[test]
    fn show_ribbon() {
        assert!(matches!(
            ok("show ribbon"),
            Command::Show { repr: RepresentationType::Ribbon, sel: None }
        ));
    }

    #[test]
    fn show_ball_stick_aliases() {
        for alias in &["ball_stick", "bs", "ball_and_stick"] {
            assert!(matches!(
                ok(&format!("show {alias}")),
                Command::Show { repr: RepresentationType::BallAndStick, .. }
            ), "alias '{alias}' failed");
        }
    }

    #[test]
    fn show_stick_aliases() {
        for alias in &["stick", "sticks"] {
            assert!(matches!(
                ok(&format!("show {alias}")),
                Command::Show { repr: RepresentationType::Stick, .. }
            ), "alias '{alias}' failed");
        }
    }

    #[test]
    fn hide_surface() {
        assert!(matches!(
            ok("hide surface"),
            Command::Hide { repr: RepresentationType::Surface, sel: None }
        ));
    }

    #[test]
    fn show_unknown_repr() {
        err("show nonsense");
    }

    // ── color ─────────────────────────────────────────────────────────────────

    #[test]
    fn color_named() {
        assert!(matches!(ok("color red"), Command::Color { spec: ColorSpec::Rgb(_), .. }));
    }

    #[test]
    fn color_element() {
        assert!(matches!(ok("color element"), Command::Color { spec: ColorSpec::Element, .. }));
        assert!(matches!(ok("color cpk"),     Command::Color { spec: ColorSpec::Element, .. }));
    }

    #[test]
    fn color_chain() {
        assert!(matches!(ok("color chain"), Command::Color { spec: ColorSpec::Chain, .. }));
    }

    #[test]
    fn color_ss() {
        assert!(matches!(ok("color ss"),        Command::Color { spec: ColorSpec::SecondaryStructure, .. }));
        assert!(matches!(ok("color secondary"), Command::Color { spec: ColorSpec::SecondaryStructure, .. }));
    }

    #[test]
    fn color_spectrum() {
        assert!(matches!(ok("color spectrum"), Command::Color { spec: ColorSpec::Spectrum, .. }));
        assert!(matches!(ok("color rainbow"),  Command::Color { spec: ColorSpec::Spectrum, .. }));
    }

    #[test]
    fn color_b_factor() {
        assert!(matches!(ok("color b"),       Command::Color { spec: ColorSpec::BFactor, .. }));
        assert!(matches!(ok("color bfactor"), Command::Color { spec: ColorSpec::BFactor, .. }));
    }

    #[test]
    fn color_unknown() {
        err("color notacolor");
    }

    #[test]
    fn color_missing_name() {
        err("color");
    }

    // ── select ────────────────────────────────────────────────────────────────

    #[test]
    fn select_basic() {
        match ok("select mysел, chain A") {
            Command::Select { name, .. } => assert_eq!(name, "mysел"),
            _ => panic!("expected Select"),
        }
    }

    // ── simple commands ───────────────────────────────────────────────────────

    #[test]
    fn quit_aliases() {
        assert!(matches!(ok("quit"), Command::Quit));
        assert!(matches!(ok("q"),    Command::Quit));
        assert!(matches!(ok("exit"), Command::Quit));
    }

    #[test]
    fn reset_cmd() {
        assert!(matches!(ok("reset"), Command::Reset));
    }

    #[test]
    fn zoom_no_sel() {
        assert!(matches!(ok("zoom"), Command::Zoom { sel: None }));
    }

    #[test]
    fn zoom_with_sel() {
        assert!(matches!(ok("zoom chain A"), Command::Zoom { sel: Some(_) }));
    }

    #[test]
    fn enable_disable() {
        assert!(matches!(ok("enable mol1"),  Command::Enable(ref n) if n == "mol1"));
        assert!(matches!(ok("disable mol1"), Command::Disable(ref n) if n == "mol1"));
    }

    #[test]
    fn delete_aliases() {
        assert!(matches!(ok("delete mol1"), Command::Delete(_)));
        assert!(matches!(ok("del mol1"),    Command::Delete(_)));
    }

    #[test]
    fn bg_command() {
        assert!(matches!(ok("bg black"),      Command::Background(_)));
        assert!(matches!(ok("background white"), Command::Background(_)));
    }

    #[test]
    fn bg_unknown_color() {
        err("bg notacolor");
    }

    // ── docktrace ─────────────────────────────────────────────────────────────

    #[test]
    fn docktrace_parse() {
        match ok("docktrace trace.tsv, ligand.pdbqt") {
            Command::DockTrace { trace_path, ligand_path } => {
                assert_eq!(trace_path.to_str().unwrap(), "trace.tsv");
                assert_eq!(ligand_path.to_str().unwrap(), "ligand.pdbqt");
            }
            _ => panic!("expected DockTrace"),
        }
    }

    #[test]
    fn docktrace_requires_two_args() {
        err("docktrace trace.tsv");
        err("docktrace");
    }

    // ── unknown command ───────────────────────────────────────────────────────

    #[test]
    fn unknown_command() {
        err("foobar");
    }

    #[test]
    fn empty_input() {
        err("");
    }
}
