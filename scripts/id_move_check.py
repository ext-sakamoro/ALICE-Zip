#!/usr/bin/env python3
"""An identifier moves only with a breaking version and naming both sides (ALICE-Law G9).

The identifiers of scripts/id_move.toml (`SEMANTICS_ID`, `*_SEMANTICS_ID`: constants a
stored value is stamped with) are read as values, not as text: most are folded by a
`const fn`, so the source can stay the same while the value moves (a resolved upstream
crate changed) and change while the value stays. For the base ref and for the working
tree, a probe binary that depends on the crate by path prints each identifier and each
previous-identifier list; the two are then compared.

The base is the newest release tag reachable from HEAD that matches a pattern of the
config (`v{version}`, `<crate>-v{version}`, ...), so an identifier moved twice between two
releases counts once, against what was published. `--base <ref>` overrides it.

For each identifier that exists at the base and has another value now:

- the version must be a breaking step from the base's (a major for 1.x and later, a
  minor or a major for 0.x);
- the base's value must be in the identifier's previous list now (data stamped with it
  stays readable as a previous identifier), when the config names one;
- CHANGELOG.md must name both values (their first 8 hex digits).

An identifier missing at the base is new and passes. Comparing no identifier fails, and
so does a config naming a constant the working tree does not have.

Completeness: every `pub const` / `pub static` of type `[u8; 32]` in the crate's src/ (the
form every identifier of these crates takes) must be an [[id]] of the config or be listed
under [exempt] with a reason, so a second identifier (a format id next to SEMANTICS_ID)
cannot be left out; an [exempt] entry that names no such constant fails, and finding none
fails as having compared nothing. A constant behind a type alias is not seen.

The base's identifiers are read from the version as published (crates.io, `={version}`),
which is what a stored value was stamped by; a version never published is read from the tag's
tree instead (`--base-from tag` forces that, `--base-from registry` refuses the fallback).

A registry that cannot be read fails the run on the default branch, on tags and locally, and
is a warning on ci/ and wip/ branches and pull requests (an outage does not turn every branch
red; the default branch still does not pass a check it did not run).

usage: scripts/id_move_check.py [--root DIR] [--base REF] [--base-from auto|registry|tag] [--print-base]
config: scripts/id_move.toml
    package = "alice-zip"                 # the cargo package
    manifest = "Cargo.toml"               # its manifest, relative to the root
    features = ["std"]                    # features the probe builds it with (optional)
    default-features = true               # (optional)
    tags = ["v{version}"]                 # release tag patterns
    [exempt]                              # [u8; 32] constants that are not identifiers
    SOME_KEY = "why it is not an identifier"
    [[id]]
    path = "alice_zip::law::SEMANTICS_ID"
    previous = "alice_zip::semantics::PREVIOUS_SEMANTICS_IDS"   # optional
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG = "scripts/id_move.toml"
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")
MISSING_RE = re.compile(r"error\[E0(?:425|432|433)\]|cannot find|unresolved import|could not find")


class CheckError(Exception):
    pass


@dataclass(frozen=True)
class IdSpec:
    path: str
    previous: str | None


@dataclass(frozen=True)
class Config:
    package: str
    manifest: str
    features: tuple[str, ...]
    default_features: bool
    tags: tuple[str, ...]
    ids: tuple[IdSpec, ...]
    exempt: tuple[tuple[str, str], ...] = ()


MIN_REASON = 12
ID_CONST_RE = re.compile(r"^\s*pub\s+(?:const|static)\s+([A-Za-z_]\w*)\s*:\s*\[\s*u8\s*;\s*32\s*\]")


def id_constants(crate_dir: Path) -> dict[str, list[str]]:
    """{name: [file:line, ...]} of every `pub const|static NAME: [u8; 32]` under src/"""
    out: dict[str, list[str]] = {}
    src = crate_dir / "src"
    for f in sorted(src.rglob("*.rs")) if src.exists() else []:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = ID_CONST_RE.match(line)
            if m:
                out.setdefault(m[1], []).append(f"{f.relative_to(crate_dir).as_posix()}:{n}")
    return out


def completeness(cfg: Config, found: dict[str, list[str]]) -> list[str]:
    problems = []
    if not found:
        return ["no `pub const NAME: [u8; 32]` found under src/ (compared nothing)"]
    listed = {spec.path.rsplit("::", 1)[1] for spec in cfg.ids}
    exempt = dict(cfg.exempt)
    for name, where in sorted(found.items()):
        if name not in listed and name not in exempt:
            problems.append(f"{name} ({', '.join(where)}): a [u8; 32] constant that is neither an [[id]] of "
                            f"{CONFIG} nor under [exempt] with a reason")
    for name, reason in sorted(exempt.items()):
        if name not in found:
            problems.append(f"[exempt] {name}: names no `pub const {name}: [u8; 32]` under src/ (stale)")
        if len(reason.strip()) < MIN_REASON:
            problems.append(f"[exempt] {name}: reason shorter than {MIN_REASON} characters")
        if name in listed:
            problems.append(f"[exempt] {name}: also an [[id]]")
    return problems


def load_config(root: Path) -> Config:
    p = root / CONFIG
    if not p.exists():
        raise CheckError(f"{CONFIG} is missing")
    doc = tomllib.loads(p.read_text(encoding="utf-8"))
    ids = []
    for e in doc.get("id", []):
        path = e.get("path")
        if not isinstance(path, str) or not re.fullmatch(r"[A-Za-z_]\w*(::[A-Za-z_]\w*)+", path):
            raise CheckError(f"{CONFIG}: [[id]] path {path!r} is not a Rust path")
        prev = e.get("previous")
        if prev is not None and (not isinstance(prev, str) or not re.fullmatch(r"[A-Za-z_]\w*(::[A-Za-z_]\w*)+", prev)):
            raise CheckError(f"{CONFIG}: [[id]] previous {prev!r} is not a Rust path")
        ids.append(IdSpec(path, prev))
    tags = doc.get("tags", [])
    if not ids:
        raise CheckError(f"{CONFIG}: no [[id]] (compared nothing)")
    if not tags or not all(isinstance(t, str) and "{version}" in t for t in tags):
        raise CheckError(f"{CONFIG}: tags must be patterns containing {{version}}")
    for key in ("package", "manifest"):
        if not isinstance(doc.get(key), str) or not doc[key]:
            raise CheckError(f"{CONFIG}: {key} is missing")
    exempt = doc.get("exempt", {})
    if not isinstance(exempt, dict) or not all(isinstance(v, str) for v in exempt.values()):
        raise CheckError(f"{CONFIG}: [exempt] must map names to reasons")
    return Config(doc["package"], doc["manifest"], tuple(doc.get("features", [])),
                  bool(doc.get("default-features", True)), tuple(tags), tuple(ids), tuple(sorted(exempt.items())))


def parse_version(v: str) -> tuple[int, int, int]:
    m = VERSION_RE.match(v)
    if not m:
        raise CheckError(f"version {v!r} is not semver")
    return int(m[1]), int(m[2]), int(m[3])


def is_breaking(old: str, new: str) -> bool:
    """`new` is a semver-breaking step from `old`: a major, or a minor while the major is 0."""
    a, b = parse_version(old), parse_version(new)
    if b[0] != a[0]:
        return b[0] > a[0]
    if a[0] == 0:
        return b[1] > a[1]
    return False


def next_breaking(v: str) -> str:
    """the smallest breaking step from `v`: the next minor in 0.x, the next major from 1.0"""
    a = parse_version(v)
    return f"0.{a[1] + 1}.0" if a[0] == 0 else f"{a[0] + 1}.0.0"


def package_version(manifest_text: str, package: str) -> str:
    doc = tomllib.loads(manifest_text)
    pkg = doc.get("package", {})
    if pkg.get("name") != package:
        raise CheckError(f"manifest names package {pkg.get('name')!r}, the config {package!r}")
    v = pkg.get("version")
    if isinstance(v, dict):
        raise CheckError("version.workspace = true is not read; give the member's version")
    if not isinstance(v, str):
        raise CheckError("the manifest has no version")
    return v


def git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise CheckError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def release_base(root: Path, cfg: Config) -> tuple[str, str]:
    """(tag, version) of the newest release tag reachable from HEAD matching a pattern."""
    candidates = []
    for tag in git(root, "tag", "--merged", "HEAD").split():
        for pat in cfg.tags:
            pre, post = pat.split("{version}", 1)
            if tag.startswith(pre) and tag.endswith(post):
                v = tag[len(pre):len(tag) - len(post) if post else None]
                if VERSION_RE.match(v):
                    candidates.append((parse_version(v), tag, v))
    # newest first; a tag counts only when the manifest at it is this package at that
    # version (a repository can carry tags of another package or of an earlier language)
    for _, tag, v in sorted(candidates, reverse=True):
        r = subprocess.run(["git", "-C", str(root), "show", f"{tag}:{cfg.manifest}"], capture_output=True, text=True)
        if r.returncode != 0:
            continue
        try:
            if package_version(r.stdout, cfg.package) == v:
                return tag, v
        except (CheckError, tomllib.TOMLDecodeError):
            continue
    raise CheckError(f"no release tag reachable from HEAD matches {list(cfg.tags)} with {cfg.package} "
                     f"at that version in {cfg.manifest} (nothing published to compare with; pass --base to name one)")


def probe_source(cfg: Config, spec: IdSpec, with_previous: bool = True) -> str:
    lines = ["fn hex(b: &[u8; 32]) -> String { b.iter().map(|x| format!(\"{:02x}\", x)).collect() }",
             "fn main() {",
             f"    let id: [u8; 32] = {spec.path};",
             "    print!(\"{{\\\"id\\\": \\\"{}\\\", \\\"previous\\\": \", hex(&id));"]
    if spec.previous and with_previous:
        lines += [f"    let prev: &[[u8; 32]] = {spec.previous};",
                  "    let p: Vec<String> = prev.iter().map(hex).collect();",
                  "    println!(\"[{}]}}\", p.iter().map(|h| format!(\"\\\"{}\\\"\", h)).collect::<Vec<_>>().join(\", \"));"]
    else:
        lines.append("    println!(\"null}}\");")
    lines.append("}")
    return "\n".join(lines) + "\n"


class NotPublished(CheckError):
    pass


class RegistryUnreachable(CheckError):
    pass


UNREACHABLE_RE = re.compile(r"failed to (?:download|get|query|update)|spurious network error|Couldn't resolve host|"
                            r"Could not resolve host|timed out|SSL|failed to fetch|network failure|HTTP status (?:5\d\d|429)")


def unreadable_base_policy(env: dict[str, str]) -> str:
    """"fail" or "warn" when the base cannot be read (the registry is unreachable): a registry
    outage must not turn every branch red, but the default branch and release tags must not
    pass a check they did not run. Local runs (no GITHUB_REF) fail."""
    ref = env.get("GITHUB_REF", "")
    if env.get("GITHUB_EVENT_NAME") in ("pull_request", "pull_request_target"):
        return "warn"
    if ref.startswith("refs/heads/ci/") or ref.startswith("refs/heads/wip/"):
        return "warn"
    return "fail"


def read_ids(source: Path | str, cfg: Config, target_dir: Path, with_previous: bool = True) -> dict[str, dict | None]:
    """{path: {"id": hex, "previous": [hex] | None} or None when the crate does not have it}, of the
    crate in the tree `source` (a Path) or of the published version `source` (a str) from the
    registry; the previous lists are read only where they are judged (the working tree): the base
    may predate them"""
    out: dict[str, dict | None] = {}
    for i, spec in enumerate(cfg.ids):
        probe = Path(tempfile.mkdtemp(prefix="id-move-probe-"))
        try:
            (probe / "src").mkdir()
            if isinstance(source, Path):
                dep = {"path": str((source / cfg.manifest).resolve().parent)}
            else:
                dep = {"version": f"={source}"}
            dep["default-features"] = cfg.default_features
            if cfg.features:
                dep["features"] = list(cfg.features)
            dep_toml = "{ " + ", ".join(f"{k} = {json.dumps(v)}" for k, v in dep.items()) + " }"
            (probe / "Cargo.toml").write_text(
                f"[package]\nname = \"id-move-probe-{i}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n"
                f"[dependencies]\n{json.dumps(cfg.package)} = {dep_toml}\n\n[workspace]\n", encoding="utf-8")
            (probe / "src" / "main.rs").write_text(probe_source(cfg, spec, with_previous), encoding="utf-8")
            env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir), RUSTFLAGS="")
            env.pop("RUSTC_WRAPPER", None)
            r = subprocess.run(["cargo", "run", "--quiet", "--manifest-path", str(probe / "Cargo.toml")],
                               capture_output=True, text=True, env=env)
            if r.returncode != 0 and isinstance(source, str) and re.search(
                    r"failed to select a version|no matching package named|could not find .* in registry", r.stderr):
                raise NotPublished(f"{cfg.package} {source} is not in the registry")
            if r.returncode != 0 and isinstance(source, str) and UNREACHABLE_RE.search(r.stderr):
                raise RegistryUnreachable(f"cannot read the base {cfg.package} {source} from the registry: "
                                          f"{r.stderr.strip()[-400:]}")
            if r.returncode != 0:
                # absent only when the error is the probe's path not resolving (a segment of
                # it named in a resolution error); any other build failure is an error, so a
                # broken base cannot pass as "the identifier is new"
                segs = spec.path.split("::")[1:]
                errs = [ln for ln in r.stderr.splitlines() if MISSING_RE.search(ln)]
                if errs and "id-move-probe" in r.stderr and any(
                        re.search(rf"`{re.escape(s)}`", ln) for ln in errs for s in segs):
                    out[spec.path] = None
                    continue
                raise CheckError(f"the probe for {spec.path} did not run against {source}: {r.stderr.strip()[-800:]}")
            line = r.stdout.strip().splitlines()[-1]
            out[spec.path] = json.loads(line)
        finally:
            shutil.rmtree(probe, ignore_errors=True)
    return out


def judge(base_version: str, head_version: str, base: dict[str, dict | None],
          head: dict[str, dict | None], changelog: str, cfg: Config) -> tuple[list[str], list[str]]:
    """(problems, lines of what was compared)"""
    problems, report = [], []
    for spec in cfg.ids:
        now, then = head.get(spec.path), base.get(spec.path)
        if now is None:
            problems.append(f"{spec.path}: not found in the working tree ({CONFIG} names it)")
            continue
        if then is None:
            report.append(f"{spec.path}: new since the base ({now['id'][:8]})")
            continue
        if now["id"] == then["id"]:
            report.append(f"{spec.path}: unchanged ({now['id'][:8]})")
            continue
        old, new = then["id"], now["id"]
        report.append(f"{spec.path}: moved {old[:8]} -> {new[:8]}")
        if not is_breaking(base_version, head_version):
            fix = next_breaking(base_version)
            if head_version == base_version:
                problems.append(f"{spec.path}: moved {old[:8]} -> {new[:8]} at {head_version}, the version of the "
                                f"release it is compared with: bump the version first (set version = \"{fix}\" in "
                                f"{cfg.manifest}) and name {old[:8]} and {new[:8]} in CHANGELOG.md")
            else:
                problems.append(f"{spec.path}: moved {old[:8]} -> {new[:8]} but the version {base_version} -> "
                                f"{head_version} is not a breaking step (a major, or a minor in 0.x): set "
                                f"version = \"{fix}\" in {cfg.manifest}")
        if spec.previous:
            if old not in (now.get("previous") or []):
                problems.append(f"{spec.path}: moved, and {spec.previous} does not hold the base value {old[:8]}")
        missing = [h[:8] for h in (old, new) if h[:8] not in changelog]
        if missing:
            problems.append(f"{spec.path}: moved, and CHANGELOG.md does not name {', '.join(missing)}")
    return problems, report


def check(root: Path, base_ref: str | None, base_from: str = "auto") -> tuple[list[str], list[str]]:
    cfg = load_config(root)
    missing = completeness(cfg, id_constants((root / cfg.manifest).resolve().parent))
    problems, report = check_values(root, base_ref, base_from)
    return missing + problems, report


def check_values(root: Path, base_ref: str | None, base_from: str = "auto") -> tuple[list[str], list[str]]:
    """`base_from`: "registry" reads the base version as published (what G9 is about: a path
    dependency of the tag's tree on a sibling repository would resolve against today's sibling,
    not the released one), "tag" builds the tag's tree, "auto" the registry and the tag when the
    version was never published"""
    cfg = load_config(root)
    head_version = package_version((root / cfg.manifest).read_text(encoding="utf-8"), cfg.package)
    if base_ref is None:
        base_ref, _ = release_base(root, cfg)
    base_version = package_version(git(root, "show", f"{base_ref}:{cfg.manifest}"), cfg.package)
    changelog_path = (root / cfg.manifest).parent / "CHANGELOG.md"
    if not changelog_path.exists():
        changelog_path = root / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    work = Path(tempfile.mkdtemp(prefix="id-move-"))
    target = Path(os.environ.get("ID_MOVE_TARGET_DIR") or (work / "target"))
    source = base_ref
    if base_from in ("registry", "auto"):
        try:
            base = read_ids(base_version, cfg, target, with_previous=False)
            head = read_ids(root, cfg, target)
            source = f"{cfg.package} {base_version} from the registry"
        except NotPublished:
            if base_from == "registry":
                shutil.rmtree(work, ignore_errors=True)
                raise
            base = None
    else:
        base = None
    if base is not None:
        shutil.rmtree(work, ignore_errors=True)
        problems, report = judge(base_version, head_version, base, head, changelog, cfg)
        report.insert(0, f"base {base_ref}: {source}, working tree {head_version}")
        return problems, report
    # the base tree sits next to the root, at the same depth, so a path dependency on a
    # sibling repository (`../ALICE-SDF`) resolves from it as it does from the root
    wt = root.resolve().parent / f".{root.resolve().name}.id-move-base-{os.getpid()}"
    try:
        git(root, "worktree", "add", "--detach", "--quiet", str(wt), base_ref)
        base = read_ids(wt, cfg, target, with_previous=False)
        head = read_ids(root, cfg, target)
    finally:
        subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(wt)], capture_output=True)
        shutil.rmtree(work, ignore_errors=True)
    problems, report = judge(base_version, head_version, base, head, changelog, cfg)
    report.insert(0, f"base {base_ref}: the tag's tree ({base_version}), working tree {head_version}")
    return problems, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--base")
    ap.add_argument("--base-from", choices=["auto", "registry", "tag"], default="auto")
    ap.add_argument("--print-base", action="store_true",
                    help="print <package>-<version> of the base release (a cache key for its .crate) and exit")
    a = ap.parse_args(argv)
    try:
        if a.print_base:
            cfg = load_config(Path(a.root))
            version = release_base(Path(a.root), cfg)[1] if a.base is None else package_version(
                git(Path(a.root), "show", f"{a.base}:{cfg.manifest}"), cfg.package)
            print(f"{cfg.package}-{version}")
            return 0
        problems, report = check(Path(a.root), a.base, a.base_from)
    except RegistryUnreachable as e:
        if unreadable_base_policy(dict(os.environ)) == "warn":
            print(f"id_move_check: WARNING (not checked on this ref): {e}", file=sys.stderr)
            return 0
        print(f"id_move_check: {e}", file=sys.stderr)
        return 1
    except CheckError as e:
        print(f"id_move_check: {e}", file=sys.stderr)
        return 1
    for line in report:
        print(f"id_move_check: {line}")
    for p in problems:
        print(f"id_move_check: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"id_move_check: {len(report) - 1} identifier(s) compared, 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
