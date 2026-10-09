#!/usr/bin/env python3
"""Check that every `<!-- claim-test: NAME -->` in the READMEs names a real test.

## Why this exists

The READMEs already carried `<!-- claim-test: ... -->` markers next to their
load-bearing claims, which tells a reader "there is a test behind this". On
2026-10-09 there was **no checker**, so nothing stopped a marker from pointing
at a test that had been renamed or deleted — and nothing noticed that the
benchmark table, the one set of numbers people actually quote, carried no marker
at all. It was overstating the compression ratio by more than two orders of
magnitude (parameters-only figures printed under the word "lossless") and every
gate in the repo was green.

Prose is the one place the compiler, clippy, the oracles and the fuzzers never
look, so claims there survive being wrong. This script is the gate for that
layer:

- **Check A** — every marker resolves to a `fn <name>` in `tests/` or `src/`.
  A marker pointing nowhere is worse than no marker: it asserts coverage that
  does not exist.
- **Check B** — `README.md` and `README_ja.md` carry the **same set** of
  markers. The translation drifting away from the original is how a corrected
  number stays wrong in one language.
- **Check C** — the files that must carry markers actually do, and the total is
  non-zero. Without this the script passes by finding nothing, which is the
  failure mode every gate in this repo is required to rule out.

## Usage

    python3 scripts/claim_check.py            # exit 1 on any problem
    python3 scripts/claim_check.py --list     # show what resolved where
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Documents that are required to carry claim markers (and to agree with each
#: other). Adding a translation here makes it part of check B.
REQUIRED_DOCS = ["README.md", "README_ja.md"]

#: The minimum number of markers a required document must carry. Set from the
#: 2026-10-09 state (9 each); it exists so that deleting markers wholesale is a
#: red build rather than a quiet pass.
MIN_MARKERS_PER_DOC = 6

MARKER_RE = re.compile(r"<!--\s*claim-test:\s*([A-Za-z_][A-Za-z0-9_]*)\s*-->")


def rust_sources() -> list[Path]:
    out: list[Path] = []
    for d in ("tests", "src", "libalice/src", "libalice/tests"):
        base = ROOT / d
        if base.is_dir():
            out.extend(sorted(base.rglob("*.rs")))
    return out


def defined_test_names(files: list[Path]) -> dict[str, Path]:
    """Every `fn <name>` in the Rust sources, mapped to where it was found."""
    fn_re = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]")
    found: dict[str, Path] = {}
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in fn_re.finditer(text):
            found.setdefault(m.group(1), f)
    return found


def markers_in(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return MARKER_RE.findall(path.read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    sources = rust_sources()
    defined = defined_test_names(sources)

    per_doc: dict[str, list[str]] = {}
    for rel in REQUIRED_DOCS:
        per_doc[rel] = markers_in(ROOT / rel)

    problems: list[str] = []

    # Check C first: a gate that compared nothing has not passed.
    total = sum(len(v) for v in per_doc.values())
    if not sources:
        problems.append("Rust の source が 0 件 (tests/ と src/ が見つからない)")
    if total == 0:
        problems.append(
            "claim-test marker が 0 件 — 検査が成立していない "
            f"(対象: {', '.join(REQUIRED_DOCS)})"
        )
    for rel, names in per_doc.items():
        if not (ROOT / rel).is_file():
            problems.append(f"{rel}: 必須の文書が存在しない")
        elif len(names) < MIN_MARKERS_PER_DOC:
            problems.append(
                f"{rel}: marker {len(names)} 件 (最低 {MIN_MARKERS_PER_DOC} 件) "
                "— 主張に対する検査が減っている"
            )

    # Check A: every marker resolves.
    for rel, names in per_doc.items():
        for name in names:
            if name not in defined:
                problems.append(
                    f"{rel}: claim-test `{name}` に対応する `fn {name}` が "
                    "tests/ src/ に無い (rename か削除で主張だけ残っている)"
                )

    # Check B: the translations agree.
    sets = {rel: set(names) for rel, names in per_doc.items() if (ROOT / rel).is_file()}
    if len(sets) > 1:
        base_rel, base_set = next(iter(sets.items()))
        for rel, s in list(sets.items())[1:]:
            only_base = sorted(base_set - s)
            only_other = sorted(s - base_set)
            for name in only_base:
                problems.append(f"{rel}: `{name}` の marker が無い ({base_rel} にはある)")
            for name in only_other:
                problems.append(f"{base_rel}: `{name}` の marker が無い ({rel} にはある)")

    if args.list:
        print(f"Rust source {len(sources)} file / fn {len(defined)} 個")
        for rel, names in per_doc.items():
            print(f"\n{rel}: marker {len(names)} 件")
            for name in names:
                where = defined.get(name)
                loc = where.relative_to(ROOT) if where else "**未解決**"
                print(f"  {name} -> {loc}")
        return 1 if problems else 0

    if problems:
        print(f"claim-check: {len(problems)} 件", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(
        f"claim-check: OK (marker {total} 件を {len(per_doc)} 文書で検査、"
        f"Rust source {len(sources)} file の fn {len(defined)} 個と突合)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
