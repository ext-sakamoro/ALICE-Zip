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
