"""reconstruct against the vectors both implementations are held to
(tests/data/residual/reconstruct_vectors.txt, computed by make_fixtures.py
with Python floats, ints and struct only).

original = generated (float64) + residual (float32) in float64, then the
original's dtype: integers refuse NaN, otherwise round half to even and
saturate to the dtype's range in the integer domain (infinities too); floats
are rounded once to the dtype, an overflow giving an infinity.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.residual_compression import (  # noqa: E402
    ResidualCompressionMethod as M,
    ResidualCompressor,
)

DATA = ROOT / "tests" / "data" / "residual"
ROWS = [l.split() for l in (DATA / "reconstruct_vectors.txt").read_text().splitlines()
        if l and not l.startswith("#")]


def test_every_dtype_has_vectors():
    assert len(ROWS) == 304
    assert len({r[0] for r in ROWS}) == 11


@pytest.mark.parametrize("dtype,gen,res,want", ROWS)
def test_reconstruct_gives_the_vector(dtype, gen, res, want):
    g = np.array([int(gen, 16)], dtype="<u8").view("<f8")
    r = np.array([int(res, 16)], dtype="<u4").view("<f4")
    c = ResidualCompressor(method=M.NONE)
    rd = c.compress_residual(r, original_dtype=dtype)
    with warnings.catch_warnings():
        # an overflow to an infinity in a float dtype is the defined result
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message="overflow encountered in cast")
        if want == "error":
            with pytest.raises(ValueError, match="NaN"):
                c.reconstruct(g, rd)
            return
        out = c.reconstruct(g, rd)
    assert out.dtype == np.dtype(dtype)
    assert out.astype(out.dtype.newbyteorder("<")).tobytes().hex() == want
