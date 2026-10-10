#!/usr/bin/env python3
"""Each published version is written in one place, or agrees where it is
written twice.

- libalice (the Python wheel built by maturin) takes its version from
  libalice/Cargo.toml: libalice/pyproject.toml must not carry a static
  `version` and must list it in `dynamic`.
- the Python package alice-zip: `alice_zip.__version__` equals the `version`
  of pyproject.toml.

A run that compared nothing fails.

    python3 scripts/version_check.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_table(text: str) -> str:
    m = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def check(root: Path) -> list:
    """Problems found under `root` (empty when everything agrees), and the
    number of comparisons made, as the last element."""
    problems = []
    compared = 0

    lib = project_table((root / "libalice" / "pyproject.toml").read_text(encoding="utf-8"))
    if re.search(r"^\s*version\s*=", lib, re.M):
        problems.append("libalice/pyproject.toml: a static version (it must come from Cargo.toml)")
    dyn = re.search(r"^\s*dynamic\s*=\s*\[([^\]]*)\]", lib, re.M)
    if not dyn or '"version"' not in dyn.group(1):
        problems.append('libalice/pyproject.toml: "version" is not in dynamic')
    compared += 1

    py = project_table((root / "pyproject.toml").read_text(encoding="utf-8"))
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', py, re.M)
    init = (root / "alice_zip" / "__init__.py").read_text(encoding="utf-8")
    n = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", init, re.M)
    if not m or not n:
        problems.append("pyproject.toml version or alice_zip.__version__ not found")
    else:
        compared += 1
        if m.group(1) != n.group(1):
            problems.append(
                f"alice_zip.__version__ {n.group(1)} != pyproject.toml version {m.group(1)}")
    return problems + [compared]


def main() -> int:
    *problems, compared = check(ROOT)
    if compared == 0:
        problems.append("compared nothing")
    if problems:
        print(f"version-check: {len(problems)} problem(s)")
        for p in problems:
            print("  " + p)
        return 1
    print(f"version-check: OK ({compared} comparisons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
