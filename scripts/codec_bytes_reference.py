#!/usr/bin/env python3
"""Independent reference for the codec bytes identifiers (src/compression.rs).

Written from the definitions, not from the Rust code's output:

- the probe inputs (`codec_probe_inputs`): their names and the SHA-256 of their bytes
- the fold (`fold_codec_outputs`) on fixed vectors

`tests/codec_bytes_golden.rs` holds the values this prints; `--check` recomputes them and
fails on a difference or when nothing was compared. `--decode <dir>` reads the files the
golden test writes with `ALICE_ZIP_DUMP_CODEC_OUTPUTS=<dir>` and decodes each with Python's
own zlib / lzma, comparing with the probe input (an encoder whose bytes are recorded must
still give a stream a different decoder reads).

Standard library only.
"""

import hashlib
import lzma
import os
import sys
import zlib

DOMAIN = b"alice-zip/codec-bytes/v1"
LEVELS = [0, 1, 6, 9]


def probe_inputs():
    lcg = bytearray()
    x = 0x12345678
    for _ in range(65536):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        lcg.append(x >> 24)
    ramp = bytes(i % 251 for i in range(4096))
    return [
        ("empty", b""),
        ("one-byte", b"a"),
        ("english", b"the quick brown fox jumps over the lazy dog. " * 64),
        ("japanese", ("準備で手が絵の具だらけになった。\n" * 48).encode("utf-8")),
        ("ramp", ramp),
        ("lcg-64k", bytes(lcg)),
    ]


def fold(codec, outputs):
    h = hashlib.sha256()

    def lp(b):
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)

    lp(DOMAIN)
    lp(codec)
    h.update(len(outputs).to_bytes(8, "big"))
    for label, data in outputs:
        lp(label.encode("utf-8"))
        lp(data)
    return h.hexdigest()


FOLD_VECTORS = [
    ("deflate", []),
    ("deflate", [("a", b"")]),
    ("lzma", [("a", b"bc")]),
    ("lzma", [("ab", b"c")]),
    ("deflate", [("x", b"\x00\x01"), ("y", b"\xff")]),
]


def lines():
    out = []
    for name, data in probe_inputs():
        out.append(f"input {name} {len(data)} {hashlib.sha256(data).hexdigest()}")
    for i, (codec, outputs) in enumerate(FOLD_VECTORS):
        out.append(f"fold {i} {fold(codec.encode(), outputs)}")
    return out


def check():
    path = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "codec_bytes", "reference.txt")
    with open(path, encoding="utf-8") as f:
        recorded = [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]
    expected = lines()
    if not expected:
        print("codec_bytes_reference: nothing compared", file=sys.stderr)
        return 1
    if recorded != expected:
        for a, b in zip(recorded, expected):
            if a != b:
                print(f"recorded {a}\nreference {b}", file=sys.stderr)
        if len(recorded) != len(expected):
            print(f"recorded {len(recorded)} lines, reference {len(expected)}", file=sys.stderr)
        return 1
    print(f"codec_bytes_reference: {len(expected)} lines agree")
    return 0


def decode(directory):
    inputs = dict(probe_inputs())
    n = 0
    for fname in sorted(os.listdir(directory)):
        codec, level, name = fname.split("__")
        with open(os.path.join(directory, fname), "rb") as f:
            data = f.read()
        if codec == "zlib":
            got = zlib.decompress(data)
        elif codec == "deflate":
            got = zlib.decompress(data, -15)
        elif codec == "lzma":
            got = lzma.decompress(data, format=lzma.FORMAT_ALONE)
        else:
            print(f"unknown codec {codec}", file=sys.stderr)
            return 1
        if got != inputs[name]:
            print(f"{fname}: decodes to other bytes", file=sys.stderr)
            return 1
        n += 1
    if n == 0:
        print("codec_bytes_reference: no outputs decoded", file=sys.stderr)
        return 1
    print(f"codec_bytes_reference: {n} outputs decode to their inputs")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        sys.exit(check())
    if len(sys.argv) == 3 and sys.argv[1] == "--decode":
        sys.exit(decode(sys.argv[2]))
    print("\n".join(lines()))
