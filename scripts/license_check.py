#!/usr/bin/env python3
"""The licence is stated the same way everywhere it is stated.

ALICE-Zip is LicenseRef-Proprietary (versions published before keep the
licence they were published under). This checks that:

- the Cargo manifests of the core crate and libalice say
  `LicenseRef-Proprietary` and `publish = false`;
- both pyproject files say `license = "LicenseRef-Proprietary"` and carry no
  `License ::` classifier (PEP 639 replaces them); the root one lists exactly
  `LICENSE-Proprietary` in `license-files`, libalice's lists nothing
  (maturin finds it, and a list makes `maturin sdist` add it twice);
- `alice_zip.__license__` is `LicenseRef-Proprietary`;
- `LICENSE-Proprietary` exists at the root and libalice carries an identical
  copy (a package cannot reach files outside its directory);
- no previous licence file (`LICENSE-APACHE`, `LICENSE-MIT`, `NOTICE`) is
  left at the root or in libalice;
- the READMEs show the LicenseRef-Proprietary badge, every `License:` line
  says LicenseRef-Proprietary, and none links a previous licence file;
- no comment in `deny.toml` describes this repository's crates as MIT,
  Apache or AGPL.

A run that compares nothing fails.

    python3 scripts/license_check.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPDX = "LicenseRef-Proprietary"
TEXT = "LICENSE-Proprietary"
OLD_FILES = ("LICENSE-APACHE", "LICENSE-MIT", "NOTICE")
BADGE = "img.shields.io/badge/license-LicenseRef--Proprietary-lightgrey.svg"
# a line stating the licence, e.g. `License: LicenseRef-Proprietary`
LICENSE_LINE = re.compile(r"^\s*(?:\*\*)?(?:License|Licence)(?:\*\*)?\s*:")
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
        package = table((root / rel).read_text(encoding="utf-8"), "package")
        expect(re.search(r"^\s*publish\s*=\s*false\s*$", package, re.M) is not None,
               f"{rel}: publish = false is missing")

    for rel in ["pyproject.toml", "libalice/pyproject.toml"]:
        project = table((root / rel).read_text(encoding="utf-8"), "project")
        lic = string_field(project, "license")
        expect(lic == SPDX, f"{rel}: license is {lic!r}, not {SPDX!r}")
        files = re.search(r"^\s*license-files\s*=\s*\[([^\]]*)\]", project, re.M)
        listed = set(re.findall(r'"([^"]+)"', files.group(1))) if files else set()
        if rel == "pyproject.toml":
            expect(listed == {TEXT},
                   f"{rel}: license-files is {sorted(listed)}, not [{TEXT!r}]")
        else:
            # maturin finds both itself; a list makes `maturin sdist` add the
            # root copies twice and fail
            expect(files is None, f"{rel}: license-files is set (maturin sdist adds the files twice)")
        expect("License ::" not in project, f"{rel}: a License :: classifier remains")

    init = (root / "alice_zip" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r"^__license__\s*=\s*['\"]([^'\"]+)['\"]", init, re.M)
    expect(bool(m) and m.group(1) == SPDX, f"alice_zip.__license__ is {m and m.group(1)!r}")

    top, copy = root / TEXT, root / "libalice" / TEXT
    expect(top.is_file(), f"{TEXT} is missing at the root")
    expect(copy.is_file() and top.is_file() and copy.read_bytes() == top.read_bytes(),
           f"libalice/{TEXT} is missing or differs from the root copy")
    for name in OLD_FILES:
        for rel in [name, f"libalice/{name}"]:
            expect(not (root / rel).exists(), f"{rel} is still present")

    deny = (root / "deny.toml").read_text(encoding="utf-8")
    comments = "\n".join(l for l in deny.splitlines() if l.lstrip().startswith("#"))
    expect(not re.search(r"(alice-zip|libalice|本 repository|this repository)[^\n]*\b(MIT|Apache|AGPL)\b", comments),
           "deny.toml: a comment still describes this repository's crates as an open licence")

    for rel in READMES:
        text = (root / rel).read_text(encoding="utf-8")
        for name in OLD_FILES:
            expect(f"]({name})" not in text, f"{rel}: links {name}")
        expect(f"]({TEXT})" in text, f"{rel}: does not link {TEXT}")
        for n, line in enumerate(text.splitlines(), 1):
            if LICENSE_LINE.match(line):
                expect(line.strip().startswith(f"License: {SPDX}"),
                       f"{rel}:{n}: the licence line does not say `License: {SPDX}`: {line.strip()!r}")
        if rel != "libalice/README.md":
            expect(BADGE in text, f"{rel}: the {SPDX} badge is missing")
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
