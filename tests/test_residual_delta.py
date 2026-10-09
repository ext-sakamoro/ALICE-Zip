"""Delta residuals keep their first value; files without it are refused.

Shared fixtures (tests/data/residual/make_fixtures.py) with
libalice/tests/residual_delta.rs:
  residual_delta_python_legacy.bin  method "delta", no base, xz   -> refused
  residual_delta_rust_legacy.bin    method "delta", base_value, LZMA alone -> read
  residual_delta2.bin               method "delta2", xz            -> read
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
    decompress_delta_differences,
)

DATA = ROOT / "tests" / "data" / "residual"
VALUES = np.array([5.0, 5.5, 6.0, 4.0], dtype=np.float32)


def read(name):
    return ResidualData.from_bytes((DATA / f"residual_{name}.bin").read_bytes())


def test_the_writer_records_delta2_and_restores_the_values():
    c = ResidualCompressor(method=ResidualCompressionMethod.DELTA)
    r = c.compress_residual(VALUES)
    assert r.method == ResidualCompressionMethod.DELTA2
    back = c.decompress_residual(ResidualData.from_bytes(r.to_bytes()))
    assert np.asarray(back, dtype=np.float32).tolist() == VALUES.tolist()


def test_a_delta_file_without_its_base_is_refused():
    with pytest.raises(ValueError, match="base"):
        ResidualCompressor().decompress_residual(read("delta_python_legacy"))


def test_the_differences_of_such_a_file_are_available_on_request():
    d = decompress_delta_differences(read("delta_python_legacy"))
    assert np.asarray(d, dtype=np.float32).tolist() == [0.0, 0.5, 0.5, -2.0]


def test_an_earlier_rust_delta_file_reads_lzma_alone():
    back = ResidualCompressor().decompress_residual(read("delta_rust_legacy"))
    assert np.asarray(back, dtype=np.float32).tolist() == VALUES.tolist()


def test_a_delta2_file_reads_xz():
    back = ResidualCompressor().decompress_residual(read("delta2"))
    assert np.asarray(back, dtype=np.float32).tolist() == VALUES.tolist()
