"""Which header keys each residual version takes (tests/data/residual/
header_gating.txt, shared with libalice/tests/residual_header_gating.rs):
every cell key x version must get the same verdict from both readers."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.residual_compression import ResidualData  # noqa: E402

DATA = ROOT / "tests" / "data" / "residual"
ROWS = [l.split() for l in (DATA / "header_gating.txt").read_text().splitlines()
        if l and not l.startswith("#")]


def test_the_table_has_every_cell():
    assert len(ROWS) == 20


@pytest.mark.parametrize("name,key,version,verdict", ROWS)
def test_the_python_reader_gives_the_verdict(name, key, version, verdict):
    data = (DATA / name).read_bytes()
    if verdict == "refuse":
        with pytest.raises(ValueError):
            ResidualData.from_bytes(data)
    else:
        ResidualData.from_bytes(data)
