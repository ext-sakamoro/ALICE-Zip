"""Delta residuals come back bit for bit; files without their base are refused.

Shared fixtures (tests/data/residual/make_fixtures.py) with
libalice/tests/residual_delta.rs:
  residual_delta_python_legacy.bin  "delta", no base, xz            -> refused
  residual_delta_rust_legacy.bin    "delta", base_value, LZMA alone -> read
  residual_bitdelta.bin             "bitdelta", xz                   -> read
bitdelta_streams.txt holds the delta streams both encoders must produce.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.residual_compression import (  # noqa: E402
    ResidualCompressionMethod,
    ResidualCompressor,
    ResidualData,
    _bit_delta_decode,
    _bit_delta_encode,
    decompress_delta_differences,
)

DATA = ROOT / "tests" / "data" / "residual"
VALUES = np.array([5.0, 5.5, 6.0, 4.0], dtype=np.float32)


def read(name):
    return ResidualData.from_bytes((DATA / f"residual_{name}.bin").read_bytes())


def bits(a):
    return np.asarray(a, dtype="<f4").view("<u4").tolist()


def round_trip(values):
    c = ResidualCompressor(method=ResidualCompressionMethod.DELTA)
    r = c.compress_residual(values, original_dtype="float32")
    assert r.method == ResidualCompressionMethod.BITDELTA
    return c.decompress_residual(ResidualData.from_bytes(r.to_bytes()))


@pytest.mark.parametrize(
    "values",
    [[1e-8, 1.0, 1e8, 3.0], [0.1, 123456.7, 0.3], [5.0, 5.5, 6.0, 4.0]],
    ids=["magnitudes", "decimal", "sterbenz"],
)
def test_values_far_apart_in_magnitude_come_back_bit_for_bit(values):
    v = np.array(values, dtype=np.float32)
    assert bits(round_trip(v)) == bits(v)


def test_special_and_random_bit_patterns_come_back_bit_for_bit():
    special = np.array([0x7FC00001, 0x3F800000, 0x7F800000, 0xFF800000, 0x80000000,
                        0x00000000, 0x00000001, 0x7F7FFFFF, 0xFFFFFFFF, 0x00000047],
                       dtype="<u4").view("<f4")
    rand = np.random.default_rng(7).integers(0, 2**32, size=10_000, dtype=np.uint32).view("<f4")
    for v in (special, rand):
        assert bits(round_trip(v)) == bits(v)


def test_the_delta_streams_equal_the_shared_reference():
    rows = [l.split(" | ") for l in (DATA / "bitdelta_streams.txt").read_text().splitlines()
            if l and not l.startswith("#")]
    assert len(rows) == 5
    for name, pattern, stream in rows:
        b = np.array([int(h, 16) for h in pattern.split()], dtype="<u4")
        assert _bit_delta_encode(b.view("<f4")).hex() == stream, name
        assert _bit_delta_decode(bytes.fromhex(stream)).view("<u4").tolist() == b.tolist(), name


def test_a_delta_file_without_its_base_is_refused():
    with pytest.raises(ValueError, match="base"):
        ResidualCompressor().decompress_residual(read("delta_python_legacy"))


def test_the_differences_of_such_a_file_are_available_on_request():
    d = decompress_delta_differences(read("delta_python_legacy"))
    assert np.asarray(d, dtype=np.float32).tolist() == [0.0, 0.5, 0.5, -2.0]


def test_an_earlier_rust_delta_file_reads_lzma_alone():
    back = ResidualCompressor().decompress_residual(read("delta_rust_legacy"))
    assert np.asarray(back, dtype=np.float32).tolist() == VALUES.tolist()


def test_a_bitdelta_file_reads_xz():
    back = ResidualCompressor().decompress_residual(read("bitdelta"))
    assert np.asarray(back, dtype=np.float32).tolist() == VALUES.tolist()


def test_the_recorded_dtype_is_returned():
    back = ResidualCompressor().decompress_residual(read("bitdelta"))
    assert back.dtype == np.float32
    f64 = ResidualCompressor().decompress_residual(read("float64_1d"))
    assert f64.dtype == np.float64


def test_a_dtype_the_writer_never_records_is_refused_when_read():
    raw = (DATA / "residual_float32_1d.bin").read_bytes()
    bad = raw.replace(b'"dtype":"float32"', b'"dtype":"complex"')
    with pytest.raises(ValueError, match="dtype"):
        ResidualData.from_bytes(bad)
