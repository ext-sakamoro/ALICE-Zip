#!/usr/bin/env python3
"""Every tracked Cargo.lock matches its manifest.

A lock file in the repository that cargo would rewrite on the next build resolves
other versions than the manifests require (an old `alice-det-math`, an old core
crate), and every build quietly repairs it in the working tree, so nothing fails.
For each tracked `Cargo.lock` this runs `cargo metadata --locked` on the manifest
next to it; cargo refuses when the lock file would change.

A run that checked no lock file fails.

usage: python3 scripts/lock_check.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def run(cmd: list[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def tracked_locks(runner: Runner) -> list[str]:
    out = runner(["git", "ls-files", "--", "Cargo.lock", "*/Cargo.lock"])
    if out.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {out.stderr.strip()}")
    return sorted(l for l in out.stdout.splitlines() if l.strip())


def check(runner: Runner) -> tuple[list[str], int]:
    """(problems, lock files checked)"""
    problems = []
    checked = 0
    for lock in tracked_locks(runner):
        manifest = str(Path(lock).parent / "Cargo.toml")
        # not --no-deps: without the dependency resolution cargo does not look at the
        # lock file at all, and a stale one passes
        res = runner(["cargo", "metadata", "--locked", "--format-version", "1",
                      "--manifest-path", manifest])
        checked += 1
        if res.returncode != 0:
            first = next((l for l in res.stderr.splitlines() if l.startswith("error")), res.stderr.strip())
            problems.append(f"{lock}: {first}")
    return problems, checked


def main() -> int:
    problems, checked = check(run)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if checked == 0:
        print("error: no tracked Cargo.lock was checked (compared nothing)", file=sys.stderr)
        return 1
    if problems:
        print(f"lock-check: {len(problems)} of {checked} lock files do not match their manifests")
        return 1
    print(f"lock-check: OK ({checked} lock files match their manifests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
