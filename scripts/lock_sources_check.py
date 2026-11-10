#!/usr/bin/env python3
"""Fail when a tracked Cargo.lock takes an alice-* crate by path from outside this repository or
from a registry, or links more than one alice-det-math

Every alice-* dependency comes by git (branch main + version): the lock records the commit. This
repository's own packages are by path. compat/v0_10_reader takes the published 0.10 on purpose (it
reads the formats 0.10 wrote) and is left out.

    python3 scripts/lock_sources_check.py            (the locks `git ls-files` lists)
    python3 scripts/lock_sources_check.py LOCK ...

Exit 0 when every lock passes, 1 on a finding, 2 when no lock was compared.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib

OWN = {"alice-zip", "alice-zip-cli", "alice-zip-fuzz", "alice-zip-enterprise", "alice-zip-wasm",
       "alice-zip-compat-v0-10-reader"}
LEFT_OUT = ("compat/",)


def findings(lock: str, text: str) -> list[str]:
    pk = [p for p in tomllib.loads(text).get("package", []) if p["name"].startswith("alice")]
    out = []
    for p in pk:
        src = p.get("source", "")
        if (not src and p["name"] not in OWN) or (src and not src.startswith("git+")):
            out.append(f"{lock}: {p['name']} {p['version']} from {src or 'a path'}")
    if sum(p["name"] == "alice-det-math" for p in pk) > 1:
        out.append(f"{lock}: more than one alice-det-math")
    return out


def main(argv: list[str]) -> int:
    locks = argv or subprocess.run(["git", "ls-files", "*Cargo.lock"], capture_output=True,
                                   text=True, check=True).stdout.split()
    locks = [lock for lock in locks if not lock.startswith(LEFT_OUT)]
    bad = [f for lock in locks for f in findings(lock, open(lock, encoding="utf-8").read())]
    print(f"lock sources: {len(locks)} lock(s) compared, {len(bad)} finding(s)")
    for f in bad:
        print(f"  {f}")
    if not locks:
        return 2
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
