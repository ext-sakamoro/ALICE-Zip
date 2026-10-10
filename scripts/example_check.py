#!/usr/bin/env python3
"""The output of `cargo run --release --example compression_ratio --features
lzma` (the source of the README's ratios) has its first table — law plus
xor residual — with every row bit-exact.

    python3 scripts/example_check.py example.txt
Fails when the table is missing, has no row, or a row is not bit-exact.
"""
import sys

EXPECTED_ROWS = 4


def check(text: str) -> list:
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if "Law + residual (xor)" in l)
    except StopIteration:
        return ["the law + xor residual table is missing"]
    header = [c.strip() for c in lines[start].strip().strip("|").split("|")]
    col = header.index("Bit-exact")
    rows = []
    for l in lines[start + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip() for c in l.strip().strip("|").split("|")])
    problems = []
    if len(rows) != EXPECTED_ROWS:
        problems.append(f"{len(rows)} rows, expected {EXPECTED_ROWS}")
    problems += [f"{r[0]}: bit-exact is {r[col]!r}" for r in rows if r[col] != "yes"]
    return problems


def main() -> int:
    problems = check(open(sys.argv[1], encoding="utf-8").read())
    if problems:
        print("example-check: " + "; ".join(problems))
        return 1
    print(f"example-check: OK ({EXPECTED_ROWS} rows bit-exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
