"""Writes the LZMA fallback `ALICE_ZIP` fixtures with the Python writer.

`ALICEZip.compress` is the writer these files come from in practice, so the
Rust reader is checked against its output rather than against bytes it made
itself. Each `.alice` file is written next to `.orig`, the raw bytes of the
array it compresses (`ndarray.tobytes()`), which the reader must return.
`legacy_lzma_bomb.alice` is derived from the uint8 file: its xz stream
expands to 400 MiB while the header states 4096 bytes. `legacy_lzma_f64_*`
are the float64 file with its data in an xz stream of another check type
(CRC-32, none, SHA-256), which the writer never uses.

Run from the repository root:

    python tests/data/container/make_legacy_lzma.py
"""

import lzma
import pathlib
import struct
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
    # Five bytes: the LZMA2 data is not a multiple of four bytes, so the
    # block carries padding.
    yield "legacy_lzma_small", np.array([148, 111, 74, 37, 0], dtype=np.uint8)


def bomb(name, file, expand):
    """The header and metadata of `file` with an xz stream that expands to
    `expand` zero bytes, far beyond the header's `original_size`. Not writer
    output: a reader must refuse it without allocating the expansion."""
    payload = file[66:]
    meta_end = 4 + struct.unpack("<I", payload[:4])[0]
    xz = lzma.compress(bytes(expand))
    new_payload = payload[:meta_end] + xz
    header = bytearray(file[:66])
    header[22:30] = struct.pack("<Q", len(new_payload))  # compressed_size
    (OUT / f"{name}.alice").write_bytes(bytes(header) + new_payload)
    print(f"{name}: {66 + len(new_payload)} bytes, expands to {expand} bytes")


def other_check(name, file, original, check):
    """`file` with its xz stream re-made from the same data with another
    check type. Not writer output (the writer uses CRC-64)."""
    payload = file[66:]
    meta_end = 4 + struct.unpack("<I", payload[:4])[0]
    new_payload = payload[:meta_end] + lzma.compress(original, check=check)
    header = bytearray(file[:66])
    header[22:30] = struct.pack("<Q", len(new_payload))
    (OUT / f"{name}.alice").write_bytes(bytes(header) + new_payload)


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
        if name == "legacy_lzma_u8":
            bomb("legacy_lzma_bomb", data, 400 << 20)
        if name == "legacy_lzma_f64":
            for check, tag in [(lzma.CHECK_CRC32, "crc32"), (lzma.CHECK_NONE, "nocheck"),
                               (lzma.CHECK_SHA256, "sha256")]:
                other_check(f"legacy_lzma_f64_{tag}", data, arr.tobytes(), check)


if __name__ == "__main__":
    main()
