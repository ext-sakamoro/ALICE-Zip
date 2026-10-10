"""Writes the LZMA fallback `ALICE_ZIP` fixtures with the Python writer.

`ALICEZip.compress` is the writer these files come from in practice, so the
Rust reader is checked against its output rather than against bytes it made
itself. Each `.alice` file is written next to `.orig`, the raw bytes of the
array it compresses (`ndarray.tobytes()`), which the reader must return.

Run from the repository root:

    python tests/data/container/make_legacy_lzma.py
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from alice_zip.core import ALICEZip, AliceFileHeader, AlicePayloadType  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
LZMA_FALLBACK = 0x30


def arrays():
    rng = np.random.default_rng(7)
    # Random bytes and random float64 values fit no generator, so the writer
    # falls back to LZMA. The second one is two dimensional.
    yield "legacy_lzma_u8", rng.integers(0, 256, 4096, dtype=np.uint8)
    yield "legacy_lzma_f64", rng.standard_normal((16, 8))
    # A zero-dimensional array (`"shape":[]`) and an empty one (`[0]`).
    yield "legacy_lzma_scalar", np.array(3.5)
    yield "legacy_lzma_empty", np.zeros((0,), dtype=np.int32)


def main():
    zipper = ALICEZip()
    for name, arr in arrays():
        data = zipper.compress(arr)
        header = AliceFileHeader.from_bytes(data)
        assert header.payload_type == AlicePayloadType.LZMA_FALLBACK, name
        assert header.payload_type.value == LZMA_FALLBACK, name
        back = zipper.decompress(data)
        assert back.dtype == arr.dtype and back.shape == arr.shape, name
        assert back.tobytes() == arr.tobytes(), name
        (OUT / f"{name}.alice").write_bytes(data)
        (OUT / f"{name}.orig").write_bytes(arr.tobytes())
        print(f"{name}: {len(data)} bytes, original {arr.nbytes} bytes")


if __name__ == "__main__":
    main()
