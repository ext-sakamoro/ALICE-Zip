#!/usr/bin/env python3
"""Writes the ResidualData fixtures both readers are checked against.

The JSON header carries the keys of both writers (Rust: original_len;
Python: shape / dtype / quant_bits) so that either reader can parse the
same bytes. Only the "version" differs between the two files.
Run from the repository root: python3 tests/data/residual/make_fixtures.py
"""
import json
import struct
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAYLOAD = zlib.compress(struct.pack("<2f", 0.5, -1.25))

# version as the writers emit it (an integer) and in two spellings no writer
# emits: a float and a string; both readers must refuse the latter two
for version, name in ((2, "v2"), (3, "v3"), (2.0, "v2_float"), ("2", "v2_string")):
    header = json.dumps(
        {"method": "zlib", "original_len": 2, "shape": [2], "dtype": "float32",
         "quant_bits": None, "version": version},
        separators=(",", ":"),
    ).encode()
    (HERE / f"residual_{name}.bin").write_bytes(struct.pack("<I", len(header)) + header + PAYLOAD)

# delta: the values [5.0, 5.5, 6.0, 4.0] as float32 deltas, LZMA-compressed
#   delta_python_legacy: the earlier Python writer (first delta 0, the base is
#     lost) -> both readers refuse it
#   delta_rust_legacy: the earlier Rust writer (first delta = the first value,
#     "base_value" recorded, LZMA "alone" format as lzma-rs writes it) -> both
#     readers read it
#   delta2: the current writers (method "delta2", first delta = first value,
#     xz format as Python's lzma.compress writes it)
import lzma  # noqa: E402

VALUES = [5.0, 5.5, 6.0, 4.0]


def _deltas(first, fmt):
    out = [first] + [b - a for a, b in zip(VALUES, VALUES[1:])]
    return lzma.compress(struct.pack(f"<{len(out)}f", *out), format=fmt)


def _write(name, header, payload):
    h = json.dumps(header, separators=(",", ":")).encode()
    (HERE / f"residual_{name}.bin").write_bytes(struct.pack("<I", len(h)) + h + payload)


_common = {"original_len": 4, "shape": [4], "dtype": "float32", "quant_bits": None, "version": 2}
_write("delta_python_legacy", {"method": "delta", **_common}, _deltas(0.0, lzma.FORMAT_XZ))
_write("delta_rust_legacy", {"method": "delta", "base_value": 5.0, **_common}, _deltas(5.0, lzma.FORMAT_ALONE))
_write("delta2", {"method": "delta2", **_common}, _deltas(5.0, lzma.FORMAT_XZ))
