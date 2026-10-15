#!/usr/bin/env python3
"""The libm half of the G1 gate: the crate depends on `libm` directly, and a
free function such as `libm::sin` is out of reach of the `f32::*` / `f64::*`
entries of clippy.toml. clippy.toml therefore bans every public libm function
by path except the exactly rounded ones this crate allows (ALLOWED below).

This script fails unless

  (i)   every public function of the libm version that Cargo.lock resolves is
        either in ALLOWED or listed as `libm::<name>` under disallowed-methods,
  (ii)  every `libm::*` entry of clippy.toml names a function that exists in
        that version (clippy only warns about an unreachable path),
  (iii) every ALLOWED name exists in that version,
  (iv)  `libm::Libm` (whose methods are the same functions) is listed under
        disallowed-types,
  (v)   something was compared: the libm source exported at least one
        function and clippy.toml has at least one `libm::` entry, and
  (vi)  with `--probe` (the default), clippy run on a throwaway crate that
        depends on the same libm version and carries an exact copy of
        clippy.toml flags each planted banned call exactly once (a direct
        call, a call through a `use ... as` alias, a function pointer and a
        `Libm::<f64>` method) and leaves the allowed control call alone.

The export list is read from libm's own `src/math/mod.rs` (the `pub use` lines
outside the `cfg_if!` blocks, which are the f16 / f128 forms a stable build
does not compile) and `src/lib.rs` (which must re-export only `math` and the
`Libm` helper; anything else is a new surface this script does not know).

usage: scripts/libm_ban_check.py [--no-probe] [--libm-src DIR] [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the operations IEEE 754 specifies exactly (one correctly rounded result), so
# every target computes the same bits; crate::math uses some of them in the
# no_std build
ALLOWED = frozenset({"ceil", "ceilf", "floor", "floorf", "round", "roundf",
                     "trunc", "truncf", "sqrt", "sqrtf", "fma", "fmaf"})
LIB_RS_REEXPORTS = {"pub use libm_helper::*;", "pub use self::math::*;"}
HELPER_TYPE = "libm::Libm"


def strip_cfg_if(text: str) -> str:
    """`text` without its `cfg_if! { ... }` blocks (brace matched)."""
    out: list[str] = []
    i = 0
    while True:
        j = text.find("cfg_if!", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k = text.find("{", j)
        if k < 0:
            raise ValueError("cfg_if! without a block")
        depth = 0
        for m in range(k, len(text)):
            if text[m] == "{":
                depth += 1
            elif text[m] == "}":
                depth -= 1
                if depth == 0:
                    break
        else:
            raise ValueError("unbalanced cfg_if! block")
        i = m + 1


def exported_functions(math_mod: str) -> list[str]:
    """Names re-exported by `pub use self::<mod>::{...};` outside cfg_if!."""
    names: list[str] = []
    for m in re.finditer(r"^\s*pub use self::[\w:]+?::(?:\{([^}]*)\}|(\w+));",
                         strip_cfg_if(math_mod), re.M):
        names += [x.strip() for x in (m.group(1) or m.group(2)).split(",") if x.strip()]
    return names


def unknown_lib_surface(lib_rs: str) -> list[str]:
    """`pub` lines of libm's lib.rs other than the two known re-exports."""
    out = []
    for raw in lib_rs.splitlines():
        s = raw.split("//", 1)[0].strip()
        if s.startswith("pub ") and s not in LIB_RS_REEXPORTS:
            out.append(s)
    return out


def clippy_toml_paths(text: str) -> dict[str, list[str]]:
    """`path = "..."` entries under each disallowed-* array of clippy.toml."""
    found: dict[str, list[str]] = {"disallowed-methods": [], "disallowed-types": []}
    current = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if current is None:
            m = re.match(r"^(disallowed-[a-z]+)\s*=\s*\[", line)
            if m and m.group(1) in found:
                current = m.group(1)
                line = line[m.end():]
            else:
                continue
        found[current] += re.findall(r'path\s*=\s*"([^"]+)"', line)
        if line.endswith("]"):
            current = None
    return found


def check(exports: list[str], surface: list[str], toml: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    banned = [p.removeprefix("libm::") for p in toml["disallowed-methods"] if p.startswith("libm::")]
    if not exports:
        problems.append("compared nothing: no public function was read from the libm source")
    if not banned:
        problems.append("compared nothing: clippy.toml has no libm:: entry under disallowed-methods")
    if problems:
        return problems
    for s in surface:
        problems.append(f"libm's lib.rs has a public item this check does not know: {s!r}")
    have = set(exports)
    for name in sorted(have - ALLOWED - set(banned)):
        problems.append(f"libm::{name} is public in the resolved libm but neither allowed nor banned in clippy.toml")
    for name in sorted(set(banned) - have):
        problems.append(f"clippy.toml bans libm::{name}, which the resolved libm does not export")
    for name in sorted(set(banned) & ALLOWED):
        problems.append(f"clippy.toml bans libm::{name}, which is in the allowed set")
    for name in sorted(ALLOWED - have):
        problems.append(f"allowed libm::{name} is not exported by the resolved libm")
    dup = sorted({b for b in banned if banned.count(b) > 1})
    if dup:
        problems.append(f"clippy.toml lists these libm functions more than once: {dup}")
    if HELPER_TYPE not in toml["disallowed-types"]:
        problems.append(f"{HELPER_TYPE} is not listed under disallowed-types in clippy.toml")
    return problems


PROBE_LIB = """\
pub fn direct(x: f64) -> f64 { libm::sin(x) }
pub fn alias(x: f32) -> f32 { use libm::cosf as c; c(x) }
pub fn pointer(x: f64) -> f64 { let f: fn(f64) -> f64 = libm::log2; f(x) }
pub fn helper(x: f64) -> f64 { libm::Libm::<f64>::atan2(x, 1.0) }
pub fn control(x: f64) -> f64 { libm::fma(libm::sqrt(x), libm::floor(x), libm::round(x)) }
"""
# (1-based line of lib.rs, the path the diagnostic must name)
PROBE_EXPECT = [(1, "libm::sin"), (2, "libm::cosf"), (3, "libm::log2"), (4, "libm::Libm")]
PROBE_CONTROL_LINE = 5


def probe_problems(exit_code: int, diags: list[tuple[int, str]]) -> list[str]:
    problems: list[str] = []
    if exit_code == 0:
        problems.append("probe: clippy exited 0 although the probe crate calls banned libm functions")
    for line, path in PROBE_EXPECT:
        hits = [m for ln, m in diags if ln == line]
        named = [m for m in hits if f"`{path}`" in m]
        if len(hits) != 1 or len(named) != 1:
            problems.append(f"probe: line {line} got {len(hits)} disallowed diagnostic(s), "
                            f"{len(named)} naming `{path}` (expected exactly 1)")
    control = [m for ln, m in diags if ln == PROBE_CONTROL_LINE]
    if control:
        problems.append(f"probe: the allowed control call got {len(control)} disallowed diagnostic(s): {control}")
    expected_lines = {ln for ln, _ in PROBE_EXPECT} | {PROBE_CONTROL_LINE}
    stray = [(ln, m) for ln, m in diags if ln not in expected_lines]
    if stray:
        problems.append(f"probe: disallowed diagnostic(s) outside the probe lines: {stray}")
    return problems


def disallowed_diagnostics(stdout: str) -> list[tuple[int, str]]:
    out = []
    for raw in stdout.splitlines():
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-message":
            continue
        d = msg.get("message") or {}
        code = (d.get("code") or {}).get("code")
        if code not in ("clippy::disallowed_methods", "clippy::disallowed_types"):
            continue
        spans = [s for s in d.get("spans", []) if s.get("is_primary")] or d.get("spans", [])
        out.append((spans[0]["line_start"] if spans else 0, d.get("message", "")))
    return out


def run_probe(root: Path, version: str) -> list[str]:
    tmp = Path(tempfile.mkdtemp(prefix="libm-ban-probe-"))
    try:
        (tmp / "src").mkdir()
        (tmp / "Cargo.toml").write_text(
            "[package]\nname = \"libm-ban-probe\"\nversion = \"0.0.0\"\nedition = \"2021\"\n"
            "publish = false\n\n[lib]\npath = \"src/lib.rs\"\n\n[workspace]\n\n"
            f"[dependencies]\nlibm = \"={version}\"\n", encoding="utf-8")
        (tmp / "src" / "lib.rs").write_text(PROBE_LIB, encoding="utf-8")
        shutil.copyfile(root / "clippy.toml", tmp / "clippy.toml")
        for name in ("rust-toolchain.toml", "rust-toolchain"):
            if (root / name).exists():
                shutil.copyfile(root / name, tmp / name)
        p = subprocess.run(["cargo", "clippy", "--lib", "--quiet", "--message-format=json", "--", "-D", "warnings"],
                           cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace")
        diags = disallowed_diagnostics(p.stdout)
        problems = probe_problems(p.returncode, diags)
        if problems and not diags:
            sys.stderr.write(p.stderr[-4000:])
        return problems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def resolved_libm(root: Path) -> tuple[Path, str]:
    """Source directory and version of the libm the crate resolves (Cargo.lock)."""
    p = subprocess.run(["cargo", "metadata", "--format-version", "1", "--locked"],
                       cwd=root, capture_output=True, text=True, encoding="utf-8", check=True)
    meta = json.loads(p.stdout)
    pkgs = [x for x in meta["packages"] if x["name"] == "libm"]
    if len(pkgs) != 1:
        raise SystemExit(f"libm_ban_check: expected exactly one libm in the resolved graph, found {len(pkgs)}")
    return Path(pkgs[0]["manifest_path"]).parent, pkgs[0]["version"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--libm-src", type=Path, help="libm source directory (default: from cargo metadata)")
    ap.add_argument("--no-probe", action="store_true", help="skip the clippy probe (no cargo run)")
    args = ap.parse_args(argv)

    version = None
    src = args.libm_src
    if src is None:
        src, version = resolved_libm(args.root)
    exports = exported_functions((src / "src" / "math" / "mod.rs").read_text(encoding="utf-8"))
    surface = unknown_lib_surface((src / "src" / "lib.rs").read_text(encoding="utf-8"))
    toml = clippy_toml_paths((args.root / "clippy.toml").read_text(encoding="utf-8"))
    problems = check(exports, surface, toml)
    probed = False
    if not problems and not args.no_probe:
        if version is None:
            problems.append("--probe needs the resolved version: drop --libm-src or pass --no-probe")
        else:
            problems += run_probe(args.root, version)
            probed = True
    if problems:
        for p in problems:
            print(f"libm_ban_check: {p}", file=sys.stderr)
        return 1
    banned = sum(1 for p in toml["disallowed-methods"] if p.startswith("libm::"))
    print(f"libm_ban_check: libm {version or src}: {len(set(exports))} public functions, "
          f"{len(ALLOWED)} allowed, {banned} banned in clippy.toml"
          + (f", probe flagged {len(PROBE_EXPECT)} of {len(PROBE_EXPECT)} with the control clean" if probed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
