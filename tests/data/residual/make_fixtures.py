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
