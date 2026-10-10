"""Lossless writing from the original: positions that generated + residual
cannot rebuild bit for bit are kept as exceptions (the original's bytes).

Position i is an exception exactly when the original or the generated value
is not finite, or when the reconstruct rule applied to generated and the
float32 residual does not give the original's bits. The residual there is 0.
The payload is the compressed residual followed by a raw block: the positions
(u64, ascending) then the original elements (the dtype's little-endian bytes).
A file with exceptions has version 3 and "exceptions": k (k > 0); a file
without any keeps version 2, so readers that know only version 2 refuse files
they could not rebuild.
"""
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.residual_compression import (  # noqa: E402
    ResidualCompressionMethod as M,
    ResidualCompressor,
    ResidualData,
    _rebuild,
)

DATA = ROOT / "tests" / "data" / "residual"
DTYPES = ["float16", "float32", "float64", "int8", "int16", "int32", "int64",
          "uint8", "uint16", "uint32", "uint64"]
LOSSLESS = [M.NONE, M.LZMA, M.ZLIB, M.BITDELTA]
# the residual is kept in float64 where float32 cannot carry the original's
# precision (version 4, "residual_dtype")
FLOAT64_RESIDUAL = {"float64", "int32", "uint32", "int64", "uint64"}

_spec = importlib.util.spec_from_file_location("wpf", DATA / "write_python_fixtures.py")
WPF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(WPF)


def le_bytes(a):
    return np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes()


def header(data):
    n = struct.unpack("<I", data[:4])[0]
    return json.loads(data[4:4 + n])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("method", LOSSLESS, ids=lambda m: m.value)
def test_the_shared_cases_come_back_bit_for_bit(dtype, method):
    original, generated = WPF.exception_inputs(dtype)
    c = ResidualCompressor(method=method)
    rd = ResidualData.from_bytes(c.compress_original(original, generated).to_bytes())
    out = c.reconstruct(generated, rd)
    assert out.dtype == np.dtype(dtype)
    assert le_bytes(out) == le_bytes(original)


@pytest.mark.parametrize("dtype", DTYPES)
def test_exceptions_are_exactly_the_positions_the_rule_cannot_rebuild(dtype):
    original, generated = WPF.exception_inputs(dtype)
    c = ResidualCompressor(method=M.NONE)
    rd = c.compress_original(original, generated)
    h = header(rd.to_bytes())
    positions = rd.exception_positions.tolist()
    rdt = "float64" if dtype in FLOAT64_RESIDUAL else "float32"
    assert h["version"] == 4 and h["residual_dtype"] == rdt
    assert h["exceptions"] == len(positions) > 0
    assert positions == sorted(set(positions))
    residual = c.decompress_residual(rd)
    assert residual.dtype == np.dtype(rdt)
    assert all(residual.view(f"<u{residual.itemsize}")[p] == 0 for p in positions)
    # the other positions rebuild from generated + residual alone; these do not,
    # or have a value that is not finite
    width = np.dtype(dtype).itemsize
    o = le_bytes(original)
    is_float = np.issubdtype(np.dtype(dtype), np.floating)
    expected = []
    for i in range(len(original)):
        if not np.isfinite(generated[i]) or (is_float and not np.isfinite(original[i])):
            expected.append(i)
            continue
        out, invalid = _rebuild(generated[i:i + 1], residual[i:i + 1], np.dtype(dtype))
        got = None if invalid[0] else le_bytes(out)
        if got != o[i * width:(i + 1) * width]:
            expected.append(i)
    assert positions == expected


def test_finite_data_the_rule_rebuilds_keeps_version_2():
    original = np.array([5.0, 5.5, 6.0, 4.0], dtype=np.float32)
    generated = np.array([4.0, 5.0, 6.5, 4.25])
    rd = ResidualCompressor(method=M.LZMA).compress_original(original, generated)
    assert header(rd.to_bytes())["version"] == 2
    assert "exceptions" not in header(rd.to_bytes())


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("dtype", DTYPES)
def test_random_bit_patterns_come_back_bit_for_bit(dtype, seed):
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    n = 2000
    original = rng.integers(0, 256, n * dt.itemsize, dtype=np.uint8).view(dt.newbyteorder("<"))
    original = original.astype(dt)
    with np.errstate(invalid="ignore", over="ignore"):
        # a signaling NaN is quieted by the conversion: the generated value
        # only has to be near the original, not equal to its bits
        as_float = original.astype(np.float64)
        near = as_float + rng.normal(0, 4, n)
    noise = rng.integers(0, 2 ** 63, n, dtype=np.uint64).view(np.float64)
    pick = rng.integers(0, 3, n)
    generated = np.where(pick == 0, near, np.where(pick == 1, noise, as_float))
    for method in (M.LZMA, M.BITDELTA):
        c = ResidualCompressor(method=method)
        rd = ResidualData.from_bytes(c.compress_original(original, generated).to_bytes())
        assert le_bytes(c.reconstruct(generated, rd)) == le_bytes(original), method


def test_quantized_cannot_keep_exceptions():
    with pytest.raises(ValueError, match="lossless"):
        ResidualCompressor(method=M.QUANTIZED).compress_original(
            np.array([1.0, np.nan], dtype=np.float32), np.array([1.0, 1.0]))


def test_a_well_formed_file_with_exceptions_rebuilds_its_original():
    rd = ResidualData.from_bytes((DATA / "residual_exc_ok.bin").read_bytes())
    out = ResidualCompressor().reconstruct(np.array([0.0, 1.0, -1.0, 4.0]), rd)
    assert out.view("<u4").tolist() == [0x40A00000, 0x7FC00001, 0x40A00000, 0xFF800000]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("writer", ["python", "rust"])
def test_both_writers_files_rebuild_the_original(dtype, writer):
    original, generated = WPF.exception_inputs(dtype)
    rd = ResidualData.from_bytes((DATA / f"{writer}_exc_{dtype}.bin").read_bytes())
    assert le_bytes(ResidualCompressor().reconstruct(generated, rd)) == le_bytes(original)


@pytest.mark.parametrize("name", [f"exc_{d}" for d in DTYPES] + ["exc_none_float32"])
def test_both_writers_write_the_same_file_before_compression(name):
    import lzma
    def parts(data):
        # version 4: the residual stream and the exceptions are compressed together
        h = header(data)
        n = 4 + struct.unpack("<I", data[:4])[0]
        assert h["version"] == 4
        body = data[n:]
        return h, (body if h["method"] == "none" else lzma.decompress(body))
    assert parts((DATA / f"python_{name}.bin").read_bytes()) == \
        parts((DATA / f"rust_{name}.bin").read_bytes())


def test_a_value_that_is_not_finite_is_an_exception_even_when_it_would_rebuild():
    # inf - 2.0 is inf and 2.0 + inf is inf again, but the rule keeps every
    # value that is not finite as it is (libalice/tests/residual_exceptions.rs)
    rd = ResidualCompressor(method=M.NONE).compress_original(
        np.array([np.inf, 1.0, -np.inf], dtype=np.float32), np.array([2.0, 1.0, -np.inf]))
    assert rd.exception_positions.tolist() == [0, 2]


@pytest.mark.parametrize("dtype", ["float64", "int64", "uint32"])
def test_values_close_to_generated_need_no_exceptions(dtype):
    # a float64 residual rebuilds them exactly (o - g is exact when they are
    # close), so nothing is kept as it is; a float32 residual could not
    rng = np.random.default_rng(3)
    if dtype == "float64":
        original = rng.normal(0, 1000, 2000)
    else:
        original = rng.integers(2 ** 31, 2 ** 32 - 1, 2000).astype(dtype)
    generated = original.astype(np.float64) + rng.normal(0, 0.2, 2000)
    rd = ResidualCompressor(method=M.LZMA).compress_original(original, generated)
    h = header(rd.to_bytes())
    assert (h["version"], h["residual_dtype"], h["exceptions"]) == (4, "float64", 0)
    out = ResidualCompressor().reconstruct(generated, ResidualData.from_bytes(rd.to_bytes()))
    assert le_bytes(out) == le_bytes(original)


def test_a_float64_file_rebuilds_its_original():
    rd = ResidualData.from_bytes((DATA / "residual_v4_ok.bin").read_bytes())
    out = ResidualCompressor().reconstruct(np.array([0.0, 1.0, -1.0, 4.0]), rd)
    assert out.view("<u8").tolist() == [0x4014000000000000, 0x7FF0000000000001,
                                        0x4014000000000000, 0x4020000000000000]
