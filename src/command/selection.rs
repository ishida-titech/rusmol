/// Subset of PyMol's selection language.
#[derive(Debug, Clone)]
pub enum SelectionExpr {
    All,
    /// `name CA` – PDB atom name (trimmed)
    Name(String),
    /// `resn ALA` – residue name
    Resn(String),
    /// `resi 1-10` or `resi 1+5+10`
    Resi(Vec<(i32, Option<i32>)>),
    /// `chain A`
    Chain(char),
    /// `elem C`
    Element(String),
    /// `hetatm`
    Hetatm,
    /// `<object_name>` used as a scope filter
    Object(String),
    And(Box<SelectionExpr>, Box<SelectionExpr>),
    Or(Box<SelectionExpr>, Box<SelectionExpr>),
    Not(Box<SelectionExpr>),
}

// ── Parser ────────────────────────────────────────────────────────────────────

pub fn parse(input: &str) -> Result<SelectionExpr, String> {
    let tokens = tokenize(input);
    let mut pos = 0usize;
    let expr = parse_or(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(format!("unexpected token: '{}'", tokens[pos]));
    }
    Ok(expr)
}

fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for c in input.chars() {
        match c {
            '(' | ')' => {
                if !cur.is_empty() { tokens.push(cur.clone()); cur.clear(); }
                tokens.push(c.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() { tokens.push(cur.clone()); cur.clear(); }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() { tokens.push(cur); }
    tokens
}

fn parse_or(tokens: &[String], pos: &mut usize) -> Result<SelectionExpr, String> {
    let mut left = parse_and(tokens, pos)?;
    while *pos < tokens.len() && tokens[*pos].eq_ignore_ascii_case("or") {
        *pos += 1;
        let right = parse_and(tokens, pos)?;
        left = SelectionExpr::Or(Box::new(left), Box::new(right));
    }
    Ok(left)
}

fn parse_and(tokens: &[String], pos: &mut usize) -> Result<SelectionExpr, String> {
    let mut left = parse_not(tokens, pos)?;
    while *pos < tokens.len() && tokens[*pos].eq_ignore_ascii_case("and") {
        *pos += 1;
        let right = parse_not(tokens, pos)?;
        left = SelectionExpr::And(Box::new(left), Box::new(right));
    }
    Ok(left)
}

fn parse_not(tokens: &[String], pos: &mut usize) -> Result<SelectionExpr, String> {
    if *pos < tokens.len() && tokens[*pos].eq_ignore_ascii_case("not") {
        *pos += 1;
        let inner = parse_not(tokens, pos)?;
        return Ok(SelectionExpr::Not(Box::new(inner)));
    }
    parse_paren(tokens, pos)
}

fn parse_paren(tokens: &[String], pos: &mut usize) -> Result<SelectionExpr, String> {
    if *pos < tokens.len() && tokens[*pos] == "(" {
        *pos += 1;
        let expr = parse_or(tokens, pos)?;
        if *pos >= tokens.len() || tokens[*pos] != ")" {
            return Err("expected ')'".into());
        }
        *pos += 1;
        return Ok(expr);
    }
    parse_atom(tokens, pos)
}

fn parse_atom(tokens: &[String], pos: &mut usize) -> Result<SelectionExpr, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of selection".into());
    }
    let tok = tokens[*pos].to_lowercase();
    *pos += 1;

    match tok.as_str() {
        "all" | "*" => Ok(SelectionExpr::All),
        "hetatm"    => Ok(SelectionExpr::Hetatm),
        "name" => {
            let v = next_token(tokens, pos, "name")?;
            Ok(SelectionExpr::Name(v))
        }
        "resn" | "resname" => {
            let v = next_token(tokens, pos, "resn")?;
            Ok(SelectionExpr::Resn(v.to_uppercase()))
        }
        "resi" | "resi_num" | "resnum" => {
            let v = next_token(tokens, pos, "resi")?;
            Ok(SelectionExpr::Resi(parse_resi_arg(&v)?))
        }
        "chain" => {
            let v = next_token(tokens, pos, "chain")?;
            let ch = v.chars().next().ok_or("chain: expected a character")?;
            Ok(SelectionExpr::Chain(ch))
        }
        "elem" | "element" => {
            let v = next_token(tokens, pos, "elem")?;
            Ok(SelectionExpr::Element(v.to_uppercase()))
        }
        other => {
            // Treat as named selection or object reference
            Ok(SelectionExpr::Object(other.to_string()))
        }
    }
}

fn next_token<'a>(tokens: &'a [String], pos: &mut usize, ctx: &str) -> Result<String, String> {
    if *pos >= tokens.len() {
        return Err(format!("{}: expected argument", ctx));
    }
    let v = tokens[*pos].clone();
    *pos += 1;
    Ok(v)
}

/// Parse `1-10+15-20` → [(1,Some(10)), (15,Some(20))]
/// Parse `1+5+10`     → [(1,None), (5,None), (10,None)]
fn parse_resi_arg(s: &str) -> Result<Vec<(i32, Option<i32>)>, String> {
    let mut result = Vec::new();
    for part in s.split('+') {
        if let Some(dash_pos) = part[1..].find('-') {
            // range: split at first '-' after position 0 (to handle negative start)
            let split = dash_pos + 1;
            let start: i32 = part[..split].parse()
                .map_err(|_| format!("invalid resi value: {}", part))?;
            let end: i32 = part[split + 1..].parse()
                .map_err(|_| format!("invalid resi value: {}", part))?;
            result.push((start, Some(end)));
        } else {
            let val: i32 = part.parse()
                .map_err(|_| format!("invalid resi value: {}", part))?;
            result.push((val, None));
        }
    }
    Ok(result)
}

// ── Evaluator ─────────────────────────────────────────────────────────────────

use crate::scene::object::MolecularObject;
use crate::structure::atom::Atom;

/// Returns true if `atom` in `obj` matches the expression.
pub fn matches(
    expr: &SelectionExpr,
    atom: &Atom,
    obj: &MolecularObject,
) -> bool {
    match expr {
        SelectionExpr::All => true,
        SelectionExpr::Hetatm => atom.is_hetatm,
        SelectionExpr::Name(n) => atom.name.trim().eq_ignore_ascii_case(n),
        SelectionExpr::Resn(n) => atom.residue.name.eq_ignore_ascii_case(n),
        SelectionExpr::Resi(ranges) => {
            let seq = atom.residue.seq_num;
            ranges.iter().any(|&(start, end)| {
                if let Some(end) = end {
                    seq >= start && seq <= end
                } else {
                    seq == start
                }
            })
        }
        SelectionExpr::Chain(ch) => atom.residue.chain == *ch,
        SelectionExpr::Element(e) => atom.element.eq_ignore_ascii_case(e),
        SelectionExpr::Object(name) => obj.name.eq_ignore_ascii_case(name),
        SelectionExpr::And(a, b) => matches(a, atom, obj) && matches(b, atom, obj),
        SelectionExpr::Or(a, b)  => matches(a, atom, obj) || matches(b, atom, obj),
        SelectionExpr::Not(inner) => !matches(inner, atom, obj),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> SelectionExpr {
        parse(s).expect(s)
    }

    fn err(s: &str) {
        assert!(parse(s).is_err(), "expected parse error for: {s}");
    }

    // ── Parse success ────────────────────────────────────────────────────────

    #[test]
    fn parse_all() {
        assert!(matches!(ok("all"), SelectionExpr::All));
        assert!(matches!(ok("*"), SelectionExpr::All));
    }

    #[test]
    fn parse_hetatm() {
        assert!(matches!(ok("hetatm"), SelectionExpr::Hetatm));
    }

    #[test]
    fn parse_chain() {
        assert!(matches!(ok("chain A"), SelectionExpr::Chain('A')));
        assert!(matches!(ok("chain B"), SelectionExpr::Chain('B')));
    }

    #[test]
    fn parse_resn() {
        assert!(matches!(ok("resn ALA"), SelectionExpr::Resn(ref n) if n == "ALA"));
    }

    #[test]
    fn parse_name() {
        assert!(matches!(ok("name CA"), SelectionExpr::Name(ref n) if n == "CA"));
    }

    #[test]
    fn parse_elem() {
        assert!(matches!(ok("elem C"), SelectionExpr::Element(ref e) if e == "C"));
    }

    #[test]
    fn parse_resi_single() {
        let expr = ok("resi 5");
        assert!(matches!(expr, SelectionExpr::Resi(ref v) if v == &[(5, None)]));
    }

    #[test]
    fn parse_resi_range() {
        let expr = ok("resi 1-10");
        assert!(matches!(expr, SelectionExpr::Resi(ref v) if v == &[(1, Some(10))]));
    }

    #[test]
    fn parse_resi_multi() {
        let expr = ok("resi 1+5+10");
        assert!(matches!(expr, SelectionExpr::Resi(ref v) if v.len() == 3));
    }

    #[test]
    fn parse_and() {
        assert!(matches!(ok("chain A and resn ALA"), SelectionExpr::And(..)));
    }

    #[test]
    fn parse_or() {
        assert!(matches!(ok("chain A or chain B"), SelectionExpr::Or(..)));
    }

    #[test]
    fn parse_not() {
        assert!(matches!(ok("not hetatm"), SelectionExpr::Not(..)));
    }

    #[test]
    fn parse_parens() {
        // Parentheses should parse without error
        ok("(chain A and resn ALA) or hetatm");
    }

    #[test]
    fn parse_complex() {
        ok("chain A and not hetatm and resi 1-100");
    }

    // ── Parse errors ─────────────────────────────────────────────────────────

    #[test]
    fn error_empty() {
        err("");
    }

    #[test]
    fn error_unmatched_paren() {
        err("(chain A");
    }

    #[test]
    fn error_trailing_token() {
        err("chain A B");
    }
}
