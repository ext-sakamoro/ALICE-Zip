#!/usr/bin/env python3
"""Every file a tracked Rust source embeds with include_bytes! / include_str!
must itself be tracked.

A fixture that exists only in a working tree (for example because a
.gitignore pattern matches it) lets the tests compile there and fail on a
clean checkout. This reads every tracked .rs file, resolves each literal
include path relative to the file, and asks git whether the target is
tracked. It fails when any target is untracked or missing, and when it
compared nothing at all.

usage: python3 scripts/include_tracked.py [repo root]
"""
import re
import subprocess
import sys
from pathlib import Path

INCLUDE = re.compile(r'include_(?:bytes|str)!\(\s*"([^"]+)"\s*\)')


def tracked(root: Path) -> set:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True
    ).stdout.decode("utf-8")
    return {p for p in out.split("\0") if p}


def check(root: Path):
    """Returns (number of include paths compared, list of problems)."""
    files = tracked(root)
    compared = 0
    problems = []
    for rel in sorted(files):
        if not rel.endswith(".rs"):
            continue
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        # over the whole text: the macro and its path may be on different lines
        for m in INCLUDE.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            compared += 1
            target = (root / rel).parent / m.group(1)
            try:
                target_rel = target.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                problems.append(f"{rel}:{lineno}: {m.group(1)} is outside the repository")
                continue
            if target_rel not in files:
                state = "untracked" if target.exists() else "missing"
                problems.append(f"{rel}:{lineno}: {target_rel} is {state}")
    return compared, problems


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    compared, problems = check(root)
    for p in problems:
        print(f"include-tracked: {p}", file=sys.stderr)
    if compared == 0:
        print("include-tracked: compared 0 include paths", file=sys.stderr)
        return 1
    if problems:
        return 1
    print(f"include-tracked: OK ({compared} include paths, all tracked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
