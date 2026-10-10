"""The Python writer's remaining rules.

compute_residual: original - generated in float64, with the NaN chosen by
rule rather than by the hardware (x86 gives inf - inf a negative NaN, arm64 a
positive one): the original's NaN with the quiet bit set; else the generated
value's NaN with the quiet bit set; else (inf - inf) 0x7FF8000000000000.

quantize_residual in ProceduralCompressionDesigner: the residual is quantized
by ResidualCompressor (round half to even), so a value is off by at most half
a step; the earlier path truncated (up to a whole step).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.analyzers import ProceduralCompressionDesigner  # noqa: E402
from alice_zip.generators import CompressionEngine  # noqa: E402
from alice_zip.residual_compression import ResidualCompressor  # noqa: E402

QUIET = 1 << 51
NAN_CASES = [
    # original bits, generated bits, expected residual bits
    (0x7FF0000000000000, 0x7FF0000000000000, 0x7FF8000000000000),  # inf - inf
    (0xFFF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000),  # -inf - -inf
    (0x7FF4000000000000, 0x3FF0000000000000, 0x7FF4000000000000 | QUIET),
    (0x3FF0000000000000, 0xFFF0000000000001, 0xFFF0000000000001 | QUIET),
    (0x7FF0000000000123, 0xFFF8000000000456, 0x7FF0000000000123 | QUIET),
    (0x7FF8000000000001, 0x7FF0000000000000, 0x7FF8000000000001),
]


def test_the_nan_of_a_residual_is_chosen_by_rule():
    o = np.array([c[0] for c in NAN_CASES], dtype="<u8").view("<f8")
    g = np.array([c[1] for c in NAN_CASES], dtype="<u8").view("<f8")
    r = ResidualCompressor().compute_residual(o, g)
    assert r.view("<u8").tolist() == [c[2] for c in NAN_CASES]


def test_finite_residuals_are_the_float64_difference():
    o = np.array([5.0, -1.5, 1e300], dtype=np.float64)
    g = np.array([4.0, 2.25, -1e300])
    r = ResidualCompressor().compute_residual(o, g)
    assert r.tolist() == [1.0, -3.75, 2e300]


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("bits", [8, 16])
def test_a_quantized_residual_is_off_by_at_most_half_a_step(bits, dtype):
    # oracle: each value is within half a quantisation step of the original,
    # plus the rounding of the residual stored as float32 and of the output
    # to its dtype (half a unit in the last place of each, per value). For
    # float32 data with a small residual range the rounding terms can be far
    # larger than the half step (then the half step adds nothing); this input
    # is chosen so the step dominates and truncation is caught
    t = np.linspace(0, 20, 2000)
    x = (np.sin(t) * 100 + np.random.default_rng(2).normal(0, 0.3, 2000)).astype(dtype)
    d = ProceduralCompressionDesigner()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = d.compress(x, enable_lossless=True, quantize_residual=bits)
        out = np.asarray(d.decompress(r))
    assert r.engine_used == CompressionEngine.PROCEDURAL
    assert not r.is_lossless
    assert out.dtype == x.dtype
    from alice_zip.generators import decompress_from_params
    g = decompress_from_params(r.generator_params).astype(np.float64)
    res = x.astype(np.float64) - g
    step = (res.max() - res.min()) / (2 ** bits - 1)
    rounding = (np.spacing(np.abs(res).astype(np.float32)).astype(np.float64) / 2
                + np.spacing(np.abs(x)).astype(np.float64) / 2)
    err = np.abs(out.astype(np.float64) - x.astype(np.float64))
    assert np.all(err <= step / 2 + rounding + 1e-12)
    # the step dominates here, so a quantizer that truncates (up to a whole
    # step) is caught
    assert step / 2 > 4 * rounding.max()
