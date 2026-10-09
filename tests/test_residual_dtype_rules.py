"""What the recorded dtype means, and what the quantizer accepts.

The dtype in a residual header is the dtype of the original. The residual is a
float difference (original - generated), so it is returned as float32 whatever
the dtype; the dtype is applied once, when `reconstruct` rebuilds the original
(round half to even, saturate to the dtype's range, then cast). Casting the
residual itself to an integer dtype truncates it and the reconstruction misses
by one in about half the values.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.residual_compression import (  # noqa: E402
    ResidualCompressionMethod as M,
    ResidualCompressor,
)

LOSSLESS = [M.NONE, M.LZMA, M.ZLIB, M.BITDELTA]
# float32 and float64 originals are not reconstructed exactly by a float32
# residual when the generated values are float64; not part of this rule
EXACT_DTYPES = ["float16", "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64"]


def original(dtype, rng):
    if dtype.startswith(("int", "uint")):
        info = np.iinfo(dtype)
        return rng.integers(max(info.min, -30000), min(info.max, 30000), 500).astype(dtype)
    return rng.normal(0, 100, 500).astype(dtype)


@pytest.mark.parametrize("dtype", EXACT_DTYPES)
@pytest.mark.parametrize("method", LOSSLESS, ids=lambda m: m.value)
def test_a_lossless_residual_reconstructs_the_original_exactly(dtype, method):
    rng = np.random.default_rng(7)
    orig = original(dtype, rng)
    gen = orig.astype(np.float64) + rng.normal(0, 3, orig.shape)
    c = ResidualCompressor(method=method)
    rd = c.compress_residual(c.compute_residual(orig, gen), original_dtype=dtype)
    out = c.reconstruct(gen, rd)
    assert out.dtype == np.dtype(dtype)
    assert int((out != orig).sum()) == 0


@pytest.mark.parametrize("dtype", EXACT_DTYPES + ["float32", "float64"])
def test_the_residual_is_returned_as_float32_values(dtype):
    wide = np.array([-36.54, 300.5, 5.5, -129.75], dtype=np.float32)
    c = ResidualCompressor(method=M.LZMA)
    out = c.decompress_residual(c.compress_residual(wide, original_dtype=dtype))
    assert out.dtype == np.float32
    assert out.view("<u4").tolist() == wide.view("<u4").tolist()


def test_reconstruct_rounds_half_to_even_and_saturates():
    c = ResidualCompressor(method=M.NONE)
    gen = np.array([10.0, 11.0, 250.0, 5.0], dtype=np.float64)
    res = np.array([0.5, 0.5, 10.0, -9.0], dtype=np.float32)
    out = c.reconstruct(gen, c.compress_residual(res, original_dtype="uint8"))
    assert out.tolist() == [10, 12, 255, 0]


@pytest.mark.parametrize("data", [np.float32, np.float64], ids=["float32", "float64"])
def test_thirty_two_bit_codes_are_exact_on_every_platform(data):
    # oracle: exact rationals rounded half to even (see the Rust test of the
    # same name); a float32 product makes the top code 2^32, outside uint32
    q = ResidualCompressor(method=M.QUANTIZED, quantization_bits=32)._quantize_residual(
        np.array([0.0, 1.0, 0.5, 0.25], dtype=data), 32)
    assert np.frombuffer(q[16:], dtype="<u4").tolist() == [0, 4294967295, 2147483648, 1073741824]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("method,bits", [(M.QUANTIZED, None), (M.LZMA, 8), (M.QUANTIZED, 16)])
def test_values_that_are_not_finite_are_not_quantized(bad, method, bits):
    c = ResidualCompressor(method=method, quantization_bits=bits)
    with pytest.raises(ValueError, match="not finite"):
        c.compress_residual(np.array([1.0, bad, 2.0], dtype=np.float32))
