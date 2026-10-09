#!/usr/bin/env python3
"""Writes residual files with the real Python writer (alice_zip), so the Rust
reader is tested against what Python actually produces.

Needs numpy. Run from the repository root:
    python tests/data/residual/write_python_fixtures.py
The Rust writer's files come from the ignored test
`residual::tests::write_rust_writer_fixtures` in libalice (run with
`cargo test --manifest-path libalice/Cargo.toml --lib -- --ignored write_rust_writer_fixtures`).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from alice_zip.residual_compression import ResidualCompressionMethod as M, ResidualCompressor  # noqa: E402

HERE = Path(__file__).resolve().parent
VALUES = np.array([5.0, 5.5, 6.0, 4.0], dtype=np.float32)
# includes signaling NaNs (0x7F800001, 0xFF800001): a float64 round trip
# quiets them, so they show whether a reader returns the stored bits
SPECIAL = np.array([0x7FC00001, 0x3F800000, 0x7F800000, 0xFF800000, 0x80000000, 0x00000000,
                    0x00000001, 0x7F7FFFFF, 0xFFFFFFFF, 0x32000000, 0x4CBEBC20,
                    0x7F800001, 0xFF800001],
                   dtype="<u4").view("<f4")


def write(name, method, data, dtype="float32"):
    r = ResidualCompressor(method=method).compress_residual(data, original_dtype=dtype)
    (HERE / f"python_{name}.bin").write_bytes(r.to_bytes())


for name, method in [("none", M.NONE), ("lzma", M.LZMA), ("zlib", M.ZLIB),
                     ("delta", M.DELTA), ("quantized", M.QUANTIZED)]:
    write(name, method, VALUES)
for name, method in [("none", M.NONE), ("lzma", M.LZMA), ("delta", M.DELTA)]:
    write(f"{name}_special", method, SPECIAL)
# originals the Rust reader cannot return (it returns float32 of one dimension)
write("lzma_float64", M.LZMA, VALUES.astype(np.float64), "float64")
write("lzma_2d", M.LZMA, VALUES.reshape(2, 2))
