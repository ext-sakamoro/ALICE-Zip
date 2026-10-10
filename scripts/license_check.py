#!/usr/bin/env python3
"""The licence is stated the same way everywhere it is stated.

From 0.9.0 (libalice 3.0.0, Python package 2.0.0) ALICE-Zip is Apache-2.0
only. This checks that:

- the Cargo manifests of the core crate and libalice say `Apache-2.0`;
- both pyproject files say `license = "Apache-2.0"` and carry no
  `License ::` classifier (PEP 639 replaces them); the root one lists
  `LICENSE-APACHE` and `NOTICE` in `license-files`, libalice's lists nothing
  (maturin finds both, and a list makes `maturin sdist` add them twice);
- `alice_zip.__license__` is `Apache-2.0`;
- `LICENSE-APACHE` and `NOTICE` exist at the root, libalice carries
  identical copies (a package cannot reach files outside its directory), and
  no `LICENSE-MIT` is left;
- `NOTICE` carries the attribution sentence;
- the READMEs show the Apache-2.0 badge and link no `LICENSE-MIT`.

A run that compares nothing fails.

    python3 scripts/license_check.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPDX = "Apache-2.0"
ATTRIBUTION = "This product includes ALICE-Zip, Copyright 2025-2026 Moroya Sakamoto"
BADGE = "img.shields.io/badge/license-Apache--2.0-green.svg"
READMES = ["README.md", "README_ja.md", "libalice/README.md"]


def table(text: str, name: str) -> str:
    m = re.search(r"^\[" + re.escape(name) + r"\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def string_field(section: str, key: str):
    m = re.search(r'^\s*' + re.escape(key) + r'\s*=\s*"([^"]*)"', section, re.M)
    return m.group(1) if m else None


def check(root: Path) -> list:
    """Problems found under `root` (empty when everything agrees), and the
    number of comparisons made, as the last element."""
    problems = []
    compared = 0

    def expect(ok: bool, message: str):
        nonlocal compared
        compared += 1
        if not ok:
            problems.append(message)

    for rel in ["Cargo.toml", "libalice/Cargo.toml"]:
        lic = string_field(table((root / rel).read_text(encoding="utf-8"), "package"), "license")
        expect(lic == SPDX, f"{rel}: license is {lic!r}, not {SPDX!r}")

    for rel in ["pyproject.toml", "libalice/pyproject.toml"]:
        project = table((root / rel).read_text(encoding="utf-8"), "project")
        lic = string_field(project, "license")
        expect(lic == SPDX, f"{rel}: license is {lic!r}, not {SPDX!r}")
        files = re.search(r"^\s*license-files\s*=\s*\[([^\]]*)\]", project, re.M)
        listed = set(re.findall(r'"([^"]+)"', files.group(1))) if files else set()
        if rel == "pyproject.toml":
            expect({"LICENSE-APACHE", "NOTICE"} <= listed,
                   f"{rel}: license-files {sorted(listed)} lacks LICENSE-APACHE or NOTICE")
        else:
            # maturin finds both itself; a list makes `maturin sdist` add the
            # root copies twice and fail
            expect(files is None, f"{rel}: license-files is set (maturin sdist adds the files twice)")
        expect("License ::" not in project, f"{rel}: a License :: classifier remains")

    init = (root / "alice_zip" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r"^__license__\s*=\s*['\"]([^'\"]+)['\"]", init, re.M)
    expect(bool(m) and m.group(1) == SPDX, f"alice_zip.__license__ is {m and m.group(1)!r}")

    for name in ["LICENSE-APACHE", "NOTICE"]:
        top, copy = root / name, root / "libalice" / name
        expect(top.is_file(), f"{name} is missing at the root")
        expect(copy.is_file() and top.is_file() and copy.read_bytes() == top.read_bytes(),
               f"libalice/{name} is missing or differs from the root copy")
    for rel in ["LICENSE-MIT", "libalice/LICENSE-MIT"]:
        expect(not (root / rel).exists(), f"{rel} is still present")

    notice = root / "NOTICE"
    expect(notice.is_file() and ATTRIBUTION in notice.read_text(encoding="utf-8"),
           "NOTICE lacks the attribution sentence")

    for rel in READMES:
        text = (root / rel).read_text(encoding="utf-8")
        expect("LICENSE-MIT" not in text, f"{rel}: links LICENSE-MIT")
        if rel != "libalice/README.md":
            expect(BADGE in text, f"{rel}: the Apache-2.0 badge is missing")
    return problems + [compared]


def main() -> int:
    *problems, compared = check(ROOT)
    if compared == 0:
        problems.append("compared nothing")
    if problems:
        print(f"license-check: {len(problems)} problem(s)")
        for p in problems:
            print("  " + p)
        return 1
    print(f"license-check: OK ({compared} comparisons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
