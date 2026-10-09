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
for _line in (DATA / "quantized_expected.txt").read_text().splitlines():
    if _line and not _line.startswith("#"):
        _name, _bits = _line.split(" | ")
        EXPECTED[_name] = [int(h, 16) for h in _bits.split()]


def rows():
    return [l.split() for l in (DATA / "acceptance.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def test_the_python_reader_accepts_exactly_the_files_the_table_lists():
    table = rows()
    assert len(table) == 39
    for name, writer, _rust, python, values in table:
        try:
            out = ResidualCompressor().decompress_residual(
                ResidualData.from_bytes((DATA / name).read_bytes()))
        except ValueError as e:
            assert python == "refuse", f"{name} ({writer}): {e}"
            continue
        assert python == "accept", f"{name} ({writer}) was read"
        if values != "-":
            # returned in the dtype the header records; the stored values are
            # float32, so a wider dtype converts back to them exactly. A float32
            # result is compared as returned (no conversion)
            recorded = ResidualData.from_bytes((DATA / name).read_bytes()).original_dtype
            assert out.dtype == np.dtype(recorded), f"{name} ({writer}): {out.dtype}"
            f32 = out if out.dtype == np.float32 else out.astype(np.float32)
            got = np.ascontiguousarray(f32).ravel().view("<u4").tolist()
            assert got == EXPECTED[values], f"{name} ({writer})"


def payload(name):
    import json
    import lzma
    import struct
    d = (DATA / name).read_bytes()
    n = struct.unpack("<I", d[:4])[0]
    return json.loads(d[4:4 + n]), lzma.decompress(d[4 + n:])


def test_both_writers_quantize_to_the_same_bytes():
    # compared before compression (xz output differs between implementations)
    for name in ("quantized", "quantized_ties"):
        (hp, py), (hr, rs) = payload(f"python_{name}.bin"), payload(f"rust_{name}.bin")
        assert hp == hr, name
        assert py == rs, name
    # round half to even: codes 0, 2, 4, 255
    assert payload("python_quantized_ties.bin")[1][16:] == bytes([0, 2, 4, 255])

