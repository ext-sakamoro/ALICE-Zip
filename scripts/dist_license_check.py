#!/usr/bin/env python3
"""Each built wheel and sdist carries LICENSE-APACHE and NOTICE and declares
`License-Expression: Apache-2.0`.

A wheel must hold both files under `*.dist-info/licenses/`, an sdist at its
top level; the metadata (`METADATA` / `PKG-INFO`) must name both as
`License-File` and carry the expression. No file given, or a pattern that
matches nothing, fails.

    python3 scripts/dist_license_check.py target/wheels/*.whl dist/*.tar.gz
"""
import glob
import sys
import tarfile
import zipfile

FILES = ("LICENSE-APACHE", "NOTICE")
EXPRESSION = "License-Expression: Apache-2.0"


def problems_in(path: str) -> list:
    if path.endswith(".whl"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            meta = [n for n in names if n.endswith(".dist-info/METADATA")]
            metadata = z.read(meta[0]).decode("utf-8") if meta else ""
        present = {n.rsplit("/", 1)[-1] for n in names if ".dist-info/licenses/" in n}
    elif path.endswith(".tar.gz"):
        with tarfile.open(path) as t:
            names = t.getnames()
            info = [n for n in names if n.count("/") == 1 and n.endswith("/PKG-INFO")]
            metadata = t.extractfile(info[0]).read().decode("utf-8") if info else ""
        present = {n.split("/", 1)[1] for n in names if n.count("/") == 1}
    else:
        return [f"{path}: not a wheel or an sdist"]
    problems = []
    if not metadata:
        problems.append(f"{path}: no metadata")
    if EXPRESSION not in metadata.splitlines():
        problems.append(f"{path}: metadata lacks '{EXPRESSION}'")
    for f in FILES:
        if f not in present:
            problems.append(f"{path}: {f} is not in the archive")
        if f"License-File: {f}" not in metadata.splitlines():
            problems.append(f"{path}: metadata lacks 'License-File: {f}'")
    return problems


def main(patterns) -> int:
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    problems = [f"{pat}: matches nothing" for pat in patterns if not glob.glob(pat)]
    for p in paths:
        problems += problems_in(p)
    if not paths:
        problems.append("checked nothing")
    if problems:
        print(f"dist-license-check: {len(problems)} problem(s)")
        for p in problems:
            print("  " + p)
        return 1
    print(f"dist-license-check: OK ({len(paths)} archives)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
