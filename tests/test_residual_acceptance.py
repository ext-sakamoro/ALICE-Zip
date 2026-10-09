"""Which residual fixtures the Python reader accepts, from the table both
readers are tested against (tests/data/residual/acceptance.txt)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.residual_compression import ResidualCompressor, ResidualData  # noqa: E402

DATA = ROOT / "tests" / "data" / "residual"


def test_the_python_reader_accepts_exactly_the_files_the_table_lists():
    rows = [l.split() for l in (DATA / "acceptance.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    assert len(rows) == 10
    for name, _rust, python in rows:
        try:
            ResidualCompressor().decompress_residual(ResidualData.from_bytes((DATA / name).read_bytes()))
            ok = True
        except ValueError:
            ok = False
        assert ok == (python == "accept"), name
