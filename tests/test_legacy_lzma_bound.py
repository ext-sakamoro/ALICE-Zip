"""The LZMA fallback reader against the fixtures the Rust reader uses
(tests/data/container/make_legacy_lzma.py): files written by ALICEZip.compress
decode to their `.orig` bytes, and a payload that expands beyond the header's
original_size is refused without the expansion being allocated."""

import lzma
import pathlib
import struct
import tracemalloc

import pytest

from alice_zip.core import ALICEZip

DATA = pathlib.Path(__file__).resolve().parent / "data" / "container"
WRITER_FILES = ["legacy_lzma_u8", "legacy_lzma_f64", "legacy_lzma_scalar",
                "legacy_lzma_empty", "legacy_lzma_small"]
PAYLOAD_AT = 66
COMPRESSED_SIZE_AT = 22


def read(name, ext):
    return (DATA / f"{name}.{ext}").read_bytes()


def with_trailing(file, extra):
    """`file` with `extra` after the xz stream, compressed_size fixed."""
    v = bytearray(file + extra)
    v[COMPRESSED_SIZE_AT:COMPRESSED_SIZE_AT + 8] = struct.pack("<Q", len(v) - PAYLOAD_AT)
    return bytes(v)


@pytest.mark.parametrize("name", WRITER_FILES)
def test_writer_files_decode_to_the_original_bytes(name):
    out = ALICEZip().decompress(read(name, "alice"))
    assert out.tobytes() == read(name, "orig")


def test_an_expansion_past_original_size_is_refused_within_a_small_bound():
    # 61 KB of xz expanding to 400 MiB under a header stating 4096 bytes.
    bomb = read("legacy_lzma_bomb", "alice")
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="expands beyond original_size"):
            ALICEZip().decompress(bomb)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The decoder's 8 MiB dictionary (the writer's preset 6, allocated
    # through Python's allocator and so traced) plus at most 4097 bytes of
    # output; reading the whole stream would allocate 400 MiB.
    assert peak < 16 * 1024 * 1024, f"peak {peak} bytes while refusing the payload"


def test_max_output_size_bounds_the_stated_size():
    zipper = ALICEZip()
    zipper.MAX_OUTPUT_SIZE = 4095
    with pytest.raises(MemoryError, match="exceeds MAX_OUTPUT_SIZE"):
        zipper.decompress(read("legacy_lzma_u8", "alice"))
    zipper.MAX_OUTPUT_SIZE = 4096
    assert zipper.decompress(read("legacy_lzma_u8", "alice")).tobytes() == read("legacy_lzma_u8", "orig")


@pytest.mark.parametrize("extra", [b"\0\0\0\0", b"junk"], ids=["stream padding", "junk"])
def test_bytes_after_the_xz_stream_are_refused(extra):
    with pytest.raises(ValueError, match="exactly one complete xz stream"):
        ALICEZip().decompress(with_trailing(read("legacy_lzma_f64", "alice"), extra))


def test_a_second_stream_is_refused():
    empty = read("legacy_lzma_empty", "alice")
    meta_len = struct.unpack("<I", empty[PAYLOAD_AT:PAYLOAD_AT + 4])[0]
    second = empty[PAYLOAD_AT + 4 + meta_len:]
    with pytest.raises(ValueError, match="exactly one complete xz stream"):
        ALICEZip().decompress(with_trailing(read("legacy_lzma_f64", "alice"), second))


@pytest.mark.parametrize("check", [lzma.CHECK_CRC32, lzma.CHECK_NONE, lzma.CHECK_SHA256],
                         ids=["crc32", "none", "sha256"])
def test_a_check_type_other_than_crc64_is_refused(check):
    # The same data and metadata as the writer's file, in a valid xz stream
    # with another check type; only the check type differs.
    f64 = read("legacy_lzma_f64", "alice")
    meta_len = struct.unpack("<I", f64[PAYLOAD_AT:PAYLOAD_AT + 4])[0]
    xz = lzma.compress(read("legacy_lzma_f64", "orig"), check=check)
    v = bytearray(f64[:PAYLOAD_AT + 4 + meta_len] + xz)
    v[COMPRESSED_SIZE_AT:COMPRESSED_SIZE_AT + 8] = struct.pack("<Q", len(v) - PAYLOAD_AT)
    with pytest.raises(ValueError, match="the writer uses CRC-64"):
        ALICEZip().decompress(bytes(v))
