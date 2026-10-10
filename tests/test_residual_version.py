"""ResidualData refuses a header version that was never written.

Same fixtures and verdicts as libalice/tests/residual_version.rs.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.residual_compression import ResidualCompressionMethod, ResidualData  # noqa: E402

DATA = ROOT / "tests" / "data" / "residual"


def test_version_2_is_read():
    r = ResidualData.from_bytes((DATA / "residual_v2.bin").read_bytes())
    assert r.method == ResidualCompressionMethod.ZLIB
    assert tuple(r.original_shape) == (2,)


def test_a_later_version_is_refused_instead_of_being_read_as_version_2():
    with pytest.raises(ValueError, match="version 5"):
        ResidualData.from_bytes((DATA / "residual_v5.bin").read_bytes())


def test_version_3_without_exceptions_is_refused():
    # version 3 is a file with exceptions (tests/test_residual_exceptions.py)
    with pytest.raises(ValueError, match="version 3 needs"):
        ResidualData.from_bytes((DATA / "residual_v3.bin").read_bytes())


@pytest.mark.parametrize("name", ["residual_v2_float.bin", "residual_v2_string.bin"])
def test_a_version_written_as_a_float_or_a_string_is_refused(name):
    # same verdict as libalice/tests/residual_version.rs
    with pytest.raises(ValueError, match="version"):
        ResidualData.from_bytes((DATA / name).read_bytes())


def test_version_0_is_refused_as_an_unsupported_version():
    # same verdict as libalice/tests/residual_version.rs
    with pytest.raises(ValueError, match="Unsupported residual header version 0"):
        ResidualData.from_bytes((DATA / "residual_v0.bin").read_bytes())
