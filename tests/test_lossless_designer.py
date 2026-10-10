"""ProceduralCompressionDesigner.compress(enable_lossless=True) gives back the
input bit for bit.

The residual is written from the original (ResidualCompressor.compress_original):
positions the generated values plus a float32 residual cannot rebuild are
kept as they are. A result from the earlier residual path (a float32
residual, not lossless) is refused unless approximate values are asked for.
"""
import lzma
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.analyzers import ProceduralCompressionDesigner  # noqa: E402
from alice_zip.generators import CompressionEngine  # noqa: E402
from alice_zip.residual_compression import ResidualData  # noqa: E402

T = np.linspace(0, 20, 2000)
INPUTS = {
    "sin": np.sin(T) * 100,
    "poly": 0.3 * T ** 3 - 2 * T ** 2 + T - 5,
    "noise": np.sin(T) * 10 + np.random.default_rng(5).normal(0, 1, 2000),
}


def round_trip(x):
    d = ProceduralCompressionDesigner()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = d.compress(x, enable_lossless=True)
        return r, np.asarray(d.decompress(r))


def same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("name", list(INPUTS))
def test_lossless_gives_back_the_input_bit_for_bit(name, dtype):
    x = INPUTS[name].astype(dtype)
    r, out = round_trip(x)
    assert r.is_lossless
    assert same_bits(out, x)


@pytest.mark.parametrize("name", ["sin", "poly"])
def test_a_procedural_result_carries_a_residual_file(name):
    r, _ = round_trip(INPUTS[name].astype(np.float32))
    assert r.engine_used == CompressionEngine.PROCEDURAL
    assert r.metadata["residual_format"] == "alice-residual"
    rd = ResidualData.from_bytes(r.residual_data)
    assert rd.original_dtype == "float32"


def test_values_that_are_not_finite_come_back():
    x = (np.sin(T) * 100).astype(np.float32)
    x[[3, 700, 1999]] = [np.nan, np.inf, -np.inf]
    x[5] = np.array([0x7F800001], dtype="<u4").view("<f4")[0]   # signaling NaN
    _, out = round_trip(x)
    assert same_bits(out, x)


@pytest.mark.parametrize("dtype", ["int16", "uint8", "int64"])
def test_integer_input_comes_back(dtype):
    x = np.round(np.sin(T) * 100 + 120).astype(dtype)
    _, out = round_trip(x)
    assert same_bits(out, x)


def test_a_two_dimensional_input_comes_back():
    x = (np.sin(T) * 100).astype(np.float32).reshape(40, 50)
    _, out = round_trip(x)
    assert same_bits(out, x)


def earlier_result():
    """A result as the earlier lossless path wrote it: a float32 residual,
    lzma-compressed, and no residual_format."""
    x = (np.sin(T) * 100).astype(np.float64)
    r, _ = round_trip(x)
    from alice_zip.generators import decompress_from_params
    regenerated = decompress_from_params(r.generator_params).astype(np.float64)
    r.residual_data = lzma.compress((x - regenerated).astype(np.float32).tobytes())
    del r.metadata["residual_format"]
    return x, r


def test_a_result_of_the_earlier_residual_path_is_refused():
    _, r = earlier_result()
    with pytest.raises(ValueError, match="not lossless"):
        ProceduralCompressionDesigner().decompress(r)


def test_its_approximate_values_are_read_when_asked_for():
    x, r = earlier_result()
    out = ProceduralCompressionDesigner().decompress(r, allow_approximate=True)
    assert np.allclose(out, x, atol=1e-4)
