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
    with pytest.raises(ValueError, match="version 3"):
        ResidualData.from_bytes((DATA / "residual_v3.bin").read_bytes())
