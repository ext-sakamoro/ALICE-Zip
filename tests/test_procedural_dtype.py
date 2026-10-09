"""A procedural payload is regenerated in the dtype the writer recorded.

The writer stores the input dtype in the payload (`params.dtype`); the
generators used to cast every result to float32, so a float64 input came back
as float32 with half the bytes `original_size` states.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.core import ALICEZip, AliceFileHeader, AlicePayloadType  # noqa: E402


@pytest.mark.parametrize(
    "arr",
    [
        (np.linspace(0, 1, 500) ** 2 * 3 + 1).astype(np.float64),
        np.sin(np.linspace(0, 10, 1000)).astype(np.float64),
        np.sin(np.linspace(0, 10, 1000)).astype(np.float32),
    ],
    ids=["poly_f64", "sin_f64", "sin_f32"],
)
def test_the_regenerated_array_has_the_original_dtype_and_size(arr):
    data = ALICEZip().compress(arr)
    header = AliceFileHeader.from_bytes(data)
    assert header.payload_type == AlicePayloadType.PROCEDURAL, "this case must take the procedural path"
    out = np.asarray(ALICEZip().decompress(data))
    assert out.dtype == arr.dtype
    assert out.shape == arr.shape
    assert out.nbytes == header.original_size


def _with_dtype(data: bytes, new: str) -> bytes:
    """Replaces the recorded dtype with a string of the same length, so the
    payload length (and compressed_size) stay right."""
    old = b'"dtype":"float32"'
    assert data.count(old) == 1
    assert len(new) == len("float32")
    return data.replace(old, b'"dtype":"' + new.encode() + b'"')


def _f32_procedural() -> bytes:
    data = ALICEZip().compress(np.sin(np.linspace(0, 10, 1000)).astype(np.float32))
    assert AliceFileHeader.from_bytes(data).payload_type == AlicePayloadType.PROCEDURAL
    return data


@pytest.mark.parametrize("new", ["complex", "<U00007", "xyzzy12", "object_"])
def test_a_dtype_the_writer_never_records_is_refused(new):
    with pytest.raises(ValueError, match="dtype"):
        ALICEZip().decompress(_with_dtype(_f32_procedural(), new))


@pytest.mark.parametrize("new", ["float16", "float64", "uint16"])
def test_a_recorded_dtype_that_disagrees_with_original_size_is_refused(new):
    # a dtype the writer can record, but 1000 elements of it are not the
    # 4000 bytes the header states
    with pytest.raises(ValueError, match="original_size"):
        ALICEZip().decompress(_with_dtype(_f32_procedural(), new))


@pytest.mark.parametrize(
    "dtype",
    ["float16", "float32", "float64", "int8", "int16", "int32", "int64",
     "uint8", "uint16", "uint32", "uint64"],
)
def test_every_dtype_the_writer_records_reads_back(dtype):
    base = np.sin(np.linspace(0, 10, 1000))
    arr = ((base + 1) * 100).astype(dtype) if dtype.startswith(("int", "uint")) else base.astype(dtype)
    data = ALICEZip().compress(arr)
    if AliceFileHeader.from_bytes(data).payload_type != AlicePayloadType.PROCEDURAL:
        pytest.skip("not the procedural path")
    out = np.asarray(ALICEZip().decompress(data))
    assert out.dtype == arr.dtype and out.nbytes == AliceFileHeader.from_bytes(data).original_size
