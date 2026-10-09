"""Which residual files the Python reader accepts, and what it returns, from
the table both readers are tested against (tests/data/residual/acceptance.txt).

The files are the real output of each writer (write_python_fixtures.py, the
Rust writer's ignored test, the Rust writer before the header keys were
aligned) plus hand-made files for header grammar."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.residual_compression import ResidualCompressor, ResidualData  # noqa: E402

DATA = ROOT / "tests" / "data" / "residual"
EXPECTED = {
    "values": np.array([5.0, 5.5, 6.0, 4.0], dtype="<f4").view("<u4").tolist(),
    "special11": [0x7FC00001, 0x3F800000, 0x7F800000, 0xFF800000, 0x80000000, 0x00000000,
                  0x00000001, 0x7F7FFFFF, 0xFFFFFFFF, 0x32000000, 0x4CBEBC20],
}
# with signaling NaNs, which a float64 round trip would quiet
EXPECTED["special"] = EXPECTED["special11"] + [0x7F800001, 0xFF800001]


def rows():
    return [l.split() for l in (DATA / "acceptance.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def test_the_python_reader_accepts_exactly_the_files_the_table_lists():
    table = rows()
    assert len(table) == 34
    for name, writer, _rust, python, values in table:
        try:
            out = ResidualCompressor().decompress_residual(
                ResidualData.from_bytes((DATA / name).read_bytes()))
        except ValueError as e:
            assert python == "refuse", f"{name} ({writer}): {e}"
            continue
        assert python == "accept", f"{name} ({writer}) was read"
        if values != "-":
            # compared as returned (no conversion), so the reader's own dtype
            # handling is what is tested
            assert out.dtype == np.float32, f"{name} ({writer}): {out.dtype}"
            got = np.ascontiguousarray(out).ravel().view("<u4").tolist()
            assert got == EXPECTED[values], f"{name} ({writer})"
