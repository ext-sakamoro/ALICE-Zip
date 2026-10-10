#!/usr/bin/env python3
"""Which PyPI project a publishing tag names, and whether its version is
the one the package declares.

Publishing tags (`.github/workflows/pypi.yml`):

- `py-alice-zip-v<version>`: the Python package `alice-zip`; `<version>`
  must equal `version` in pyproject.toml
- `py-libalice-v<version>`: the wheel `libalice`; `<version>` must equal
  `version` in libalice/Cargo.toml (maturin takes the wheel's version from
  there)

The tags `v<version>` and `alice-zip-cli-v<version>` name the Rust crates on
crates.io and publish nothing to PyPI. Any other tag, or a version that
differs, fails; a run that compared nothing fails. On success it prints the
project name (`alice-zip` or `libalice`) on the last line.

    python3 scripts/pypi_tag_check.py py-libalice-v3.0.0
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREFIXES = {
    "py-alice-zip-v": ("alice-zip", "pyproject.toml", "project"),
    "py-libalice-v": ("libalice", "libalice/Cargo.toml", "package"),
}


def table_version(text: str, table: str) -> str | None:
    m = re.search(rf"^\[{table}\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if not m:
        return None
    v = re.search(r'^\s*version\s*=\s*"([^"]+)"', m.group(1), re.M)
    return v.group(1) if v else None


def check(tag: str, root: Path) -> tuple[str | None, list[str], int]:
    """(project, problems, comparisons)."""
    for prefix, (project, rel, table) in PREFIXES.items():
        if tag.startswith(prefix):
            want = tag[len(prefix) :]
            if not re.fullmatch(r"\d+\.\d+\.\d+([.-]?[0-9A-Za-z.]+)?", want):
                return project, [f"{tag}: '{want}' is not a version"], 0
            have = table_version((root / rel).read_text(encoding="utf-8"), table)
            if have is None:
                return project, [f"{rel}: no version in [{table}]"], 0
            if have != want:
                return project, [f"{tag}: the tag says {want}, {rel} says {have}"], 1
            return project, [], 1
    return None, [f"{tag}: not a PyPI publishing tag (py-alice-zip-v* / py-libalice-v*)"], 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    project, problems, compared = check(argv[1], ROOT)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if problems:
        return 1
    if compared == 0:
        print("error: compared nothing", file=sys.stderr)
        return 1
    print(f"pypi-tag-check: OK ({compared} comparison)")
    print(project)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
