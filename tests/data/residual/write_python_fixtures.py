#!/usr/bin/env python3
"""Writes residual files with the real Python writer (alice_zip), so the Rust
reader is tested against what Python actually produces.

Needs numpy. Importing the module computes the files (`FILES`, name -> bytes);
running it writes them. Run from the repository root:
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


FILES = {}


def write(name, method, data, dtype="float32", bits=None):
    r = ResidualCompressor(method=method, quantization_bits=bits).compress_residual(
        data, original_dtype=dtype)
    FILES[f"python_{name}.bin"] = r.to_bytes()


for name, method in [("none", M.NONE), ("lzma", M.LZMA), ("zlib", M.ZLIB),
                     ("delta", M.DELTA), ("quantized", M.QUANTIZED)]:
    write(name, method, VALUES)
for name, method in [("none", M.NONE), ("lzma", M.LZMA), ("delta", M.DELTA)]:
    write(f"{name}_special", method, SPECIAL)
# originals the Rust reader cannot return (it returns float32 of one dimension)
write("lzma_float64", M.LZMA, VALUES.astype(np.float64), "float64")
write("lzma_2d", M.LZMA, VALUES.reshape(2, 2))
# quantized: 16 bits, and quantization applied under another method (the
# writer quantizes whenever quantization_bits is set)
write("quantized16", M.QUANTIZED, VALUES, bits=16)
write("lzma_q8", M.LZMA, VALUES, bits=8)
# codes that land on .5 with an even integer below (2.5 -> 2, 4.5 -> 4 under
# round half to even, 3 and 5 under round half away from zero)
TIES = np.array([0.0, 2.5, 4.5, 255.0], dtype=np.float32)
write("quantized_ties", M.QUANTIZED, TIES)

# residuals of originals of every dtype the writer records: the residual is a
# float difference, so values outside an integer dtype's range and fractions
# are stored as they are (the dtype is that of the original, applied when the
# original is reconstructed)
WIDE = np.array([-36.54, 300.5, 5.5, -129.75], dtype=np.float32)
DTYPES = ["float16", "float32", "float64", "int8", "int16", "int32", "int64",
          "uint8", "uint16", "uint32", "uint64"]
for dt in DTYPES:
    write(f"dtype_{dt}", M.LZMA, WIDE, dt)
# 32-bit codes: 1.0 lands on the top code 4294967295, which float32 cannot hold
Q32 = np.array([0.0, 1.0, 0.5, 0.25], dtype=np.float32)
write("quantized32", M.QUANTIZED, Q32, bits=32)
# random values (the same 64 values in both writers, residual_values.random_values)
sys.path.insert(0, str(HERE))
from residual_values import random_values  # noqa: E402

RANDOM = np.array(random_values(), dtype=np.float32)
for _bits in (8, 16, 32):
    write(f"quantized_rand{_bits}", M.QUANTIZED, RANDOM, bits=_bits)


# lossless writing from the original (exceptions): the shared inputs of
# exception_cases.txt, by dtype
CASES = {}
for _line in (HERE / "exception_cases.txt").read_text().splitlines():
    if _line and not _line.startswith("#"):
        _dt, _orig, _gen = _line.split()
        CASES.setdefault(_dt, ([], []))
        CASES[_dt][0].append(bytes.fromhex(_orig))
        CASES[_dt][1].append(int(_gen, 16))


def exception_inputs(dtype):
    """(original array of dtype, generated float64 array) of one dtype."""
    orig, gen = CASES[dtype]
    original = np.frombuffer(b"".join(orig), dtype=np.dtype(dtype).newbyteorder("<"))
    generated = np.array(gen, dtype="<u8").view("<f8")
    return original.astype(np.dtype(dtype)), generated


def write_original(name, method, dtype):
    original, generated = exception_inputs(dtype)
    r = ResidualCompressor(method=method).compress_original(original, generated)
    FILES[f"python_{name}.bin"] = r.to_bytes()


for _dt in CASES:
    write_original(f"exc_{_dt}", M.LZMA, _dt)
write_original("exc_none_float32", M.NONE, "float32")

if __name__ == "__main__":
    for _name, _bytes in FILES.items():
        (HERE / _name).write_bytes(_bytes)
