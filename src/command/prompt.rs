use crossbeam_channel::{Receiver, Sender};
use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper};
use winit::event_loop::EventLoopProxy;

use super::parser::parse_command;
use super::{Command, CommandResponse, TraceAction};

// ── Tab-completion helper ────────────────────────────────────────────────────

struct RusmolHelper {
    file_completer: FilenameCompleter,
}

impl RusmolHelper {
    fn new() -> Self {
        Self {
            file_completer: FilenameCompleter::new(),
        }
    }
}

impl Helper for RusmolHelper {}
impl Hinter for RusmolHelper {
    type Hint = String;
}
impl Highlighter for RusmolHelper {}
impl Validator for RusmolHelper {}

/// Top-level command names (canonical forms only — no short aliases to avoid noise).
const COMMANDS: &[&str] = &[
    "load", "select", "show", "hide", "color", "enable", "disable", "delete",
    "zoom", "reset", "bg", "light", "light2", "set", "get", "docktrace",
    "help", "quit",
];

/// Representation names for show/hide.
const REPS: &[&str] = &[
    "ball_stick", "stick", "backbone", "ribbon", "surface", "lines",
];

/// Color names for `color` and `bg`.
const COLORS: &[&str] = &[
    "element", "chain", "ss", "spectrum", "b",
    "red", "green", "blue", "white", "black", "yellow", "orange",
    "purple", "magenta", "cyan", "grey", "pink", "salmon", "wheat",
    "teal", "marine", "forest", "limon",
];

/// Setting names for `set` / `get`.
const SETTINGS: &[&str] = &[
    "transparency", "surface_type", "surface_quality", "surface_smooth", "surface_color",
    "cartoon_color", "edge_strength", "roughness", "metallic",
    "ibl_intensity", "shadow_strength", "bloom_threshold", "bloom_intensity",
    "light_intensity", "light_elevation", "light_azimuth",
    "light2_intensity", "light2_elevation", "light2_azimuth",
];

/// Light sub-parameters.
const LIGHT_PARAMS: &[&str] = &["intensity", "elevation", "azimuth"];

/// Collect candidates from `choices` that start with `prefix`, returning
/// (replacement_start, Vec<Pair>).  `start` is the byte offset in the line
/// where the token being completed begins.
fn prefix_matches(start: usize, prefix: &str, choices: &[&str]) -> (usize, Vec<Pair>) {
    let lower = prefix.to_lowercase();
    let matches: Vec<Pair> = choices
        .iter()
        .filter(|c| c.starts_with(&lower))
        .map(|c| Pair {
            display: c.to_string(),
            replacement: c.to_string(),
        })
        .collect();
    (start, matches)
}

impl Completer for RusmolHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let before = &line[..pos];
        let trimmed = before.trim_start();

        // Split into first word (command) and the rest.
        let first_space = trimmed.find(|c: char| c.is_whitespace());

        match first_space {
            None => {
                // Still typing the command name.
                Ok(prefix_matches(before.len() - trimmed.len(), trimmed, COMMANDS))
            }
            Some(sp) => {
                let cmd = trimmed[..sp].to_lowercase();
                let rest = &before[before.len() - trimmed.len() + sp..];

                match cmd.as_str() {
                    // ── load: file path completion ──────────────────────────
                    "load" => self.file_completer.complete(line, pos, ctx),

                    // ── show / hide: representation names ──────────────────
                    "show" | "hide" => {
                        let token = last_token(rest);
                        let start = pos - token.len();
                        Ok(prefix_matches(start, token, REPS))
                    }

                    // ── color / bg: color names ────────────────────────────
                    "color" | "colour" | "bg" | "background" | "bgcolor" => {
                        let token = last_token(rest);
                        let start = pos - token.len();
                        Ok(prefix_matches(start, token, COLORS))
                    }

                    // ── set: setting name (before comma) or value ──────────
                    "set" => complete_set(rest, pos),

                    // ── get: setting names ─────────────────────────────────
                    "get" => {
                        let token = last_token(rest);
                        let start = pos - token.len();
                        Ok(prefix_matches(start, token, SETTINGS))
                    }

                    // ── light / light2: sub-parameters ─────────────────────
                    "light" | "light2" => {
                        let token = last_token(rest);
                        // Only complete parameter names, not numeric values.
                        if token.is_empty() || token.chars().next().map_or(false, |c| c.is_alphabetic()) {
                            let start = pos - token.len();
                            Ok(prefix_matches(start, token, LIGHT_PARAMS))
                        } else {
                            Ok((pos, vec![]))
                        }
                    }

                    _ => Ok((pos, vec![])),
                }
            }
        }
    }
}

/// For `set`, complete the setting name before the first comma,
/// or the value after the comma for known enum-like settings.
fn complete_set(rest: &str, pos: usize) -> rustyline::Result<(usize, Vec<Pair>)> {
    // Check if we're past the first comma (i.e. completing the value).
    if let Some(comma_pos) = rest.find(',') {
        let before_comma = rest[..comma_pos].trim();
        let after_comma = rest[comma_pos + 1..].trim_start();
        let token = last_token_str(after_comma);
        let start = pos - token.len();

        // Provide value completions for enum-like settings.
        let name = before_comma.trim().to_lowercase();
        let choices: &[&str] = match name.as_str() {
            "surface_type" => &["gaussian", "ses"],
            "surface_color" | "cartoon_color" | "ribbon_color" => {
                return Ok(prefix_matches(start, token, COLORS));
            }
            _ => return Ok((pos, vec![])),
        };
        return Ok(prefix_matches(start, token, choices));
    }

    // Before comma: complete setting name.
    let token = last_token(rest);
    let start = pos - token.len();
    Ok(prefix_matches(start, token, SETTINGS))
}

/// Get the last whitespace-delimited token from a string slice.
fn last_token(s: &str) -> &str {
    match s.rfind(|c: char| c.is_whitespace() || c == ',') {
        Some(i) => s[i + 1..].trim_start(),
        None => s.trim_start(),
    }
}

/// Same as last_token but works on a &str that may be a sub-slice.
fn last_token_str(s: &str) -> &str {
    last_token(s)
}

// ── Prompt loop ──────────────────────────────────────────────────────────────

/// Run the interactive prompt loop on a worker thread.
/// Blocks until EOF, Ctrl-C, or the main thread drops the channel.
pub fn run_prompt(cmd_tx: Sender<Command>, resp_rx: Receiver<CommandResponse>, proxy: EventLoopProxy<()>) {
    let mut rl = match Editor::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("rustyline init error: {e}");
            return;
        }
    };
    rl.set_helper(Some(RusmolHelper::new()));

    let mut trace_mode = false;
    let mut trace_step = 0usize;
    let mut trace_total = 0usize;

    loop {
        let prompt = if trace_mode {
            format!("[trace {}/{}] n/r/q> ", trace_step + 1, trace_total)
        } else {
            "RusMol> ".to_string()
        };

        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() && !trace_mode {
                    continue;
                }
                if !line.is_empty() {
                    let _ = rl.add_history_entry(&line);
                }

                if trace_mode {
                    let cmd = match line.as_str() {
                        "" | "n" | "next" => Some(Command::DockTraceNav(TraceAction::Next)),
                        "r" | "prev" => Some(Command::DockTraceNav(TraceAction::Prev)),
                        "q" | "quit" => Some(Command::DockTraceNav(TraceAction::Quit)),
                        s if s.parse::<usize>().is_ok() => {
                            let row = s.parse::<usize>().unwrap();
                            Some(Command::DockTraceNav(TraceAction::GoTo(row)))
                        }
                        _ => {
                            eprintln!("trace mode: Enter/n=next, r=prev, <number>=goto row, q=quit");
                            continue;
                        }
                    };
                    if let Some(cmd) = cmd {
                        if cmd_tx.send(cmd).is_err() { break; }
                        let _ = proxy.send_event(());
                        match resp_rx.recv() {
                            Ok(CommandResponse::DockTraceStep { step, total, info }) => {
                                trace_step = step;
                                trace_total = total;
                                println!("{info}");
                            }
                            Ok(CommandResponse::DockTraceExit) => {
                                trace_mode = false;
                                println!("dock trace mode ended");
                            }
                            Ok(CommandResponse::Ok(msg)) => {
                                if !msg.is_empty() { println!("{msg}"); }
                            }
                            Ok(CommandResponse::Error(msg)) => {
                                eprintln!("Error: {msg}");
                            }
                            Err(_) => break,
                        }
                    }
                    continue;
                }

                match parse_command(&line) {
                    Ok(cmd) => {
                        let is_quit = matches!(cmd, Command::Quit);
                        if cmd_tx.send(cmd).is_err() {
                            break; // main thread gone
                        }
                        let _ = proxy.send_event(());
                        // Wait for response from main thread
                        match resp_rx.recv() {
                            Ok(CommandResponse::Ok(msg)) => {
                                if !msg.is_empty() {
                                    println!("{msg}");
                                }
                            }
                            Ok(CommandResponse::Error(msg)) => {
                                eprintln!("Error: {msg}");
                            }
                            Ok(CommandResponse::DockTraceStep { step, total, info }) => {
                                trace_mode = true;
                                trace_step = step;
                                trace_total = total;
                                println!("{info}");
                            }
                            Ok(CommandResponse::DockTraceExit) => {
                                trace_mode = false;
                            }
                            Err(_) => break, // main thread gone
                        }
                        if is_quit {
                            break;
                        }
                    }
                    Err(e) => eprintln!("Parse error: {e}"),
                }
            }
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                let _ = cmd_tx.send(Command::Quit);
                let _ = proxy.send_event(());
                break;
            }
            Err(e) => {
                eprintln!("Readline error: {e}");
                break;
            }
        }
    }
}
