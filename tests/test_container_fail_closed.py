"""The Python ALICE_ZIP reader refuses what the writer never produced and
checks the original where the payload is lossless.

Fixtures come from tests/data/container/container_ref.py. The table of
payloads whose original_hash can be checked is read from the Rust oracle
(tests/container_oracle.rs, `CHECKABLE`) so that both readers are held to the
same rule.
"""
import hashlib
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip.core import (  # noqa: E402
    ALICEZip,
    AliceFileHeader,
    original_hash_checkable,
)

DATA = ROOT / "tests" / "data" / "container"
V1 = (DATA / "legacy_v1.alice").read_bytes()
V2 = (DATA / "legacy_v2.alice").read_bytes()
MAJOR2 = (DATA / "legacy_major2.alice").read_bytes()
UNKNOWN_PAYLOAD = (DATA / "legacy_unknown_payload.alice").read_bytes()


def test_files_the_writer_produced_are_still_read():
    h1 = AliceFileHeader.from_bytes(V1)
    assert (h1.version_major, h1.version_minor, h1.header_size()) == (1, 0, 65)
    h2 = AliceFileHeader.from_bytes(V2)
    assert (h2.version_major, h2.version_minor, h2.header_size()) == (1, 1, 66)
    assert h2.compressed_size == len(V2) - 66


@pytest.mark.parametrize(
    "data",
    [
        MAJOR2,
        UNKNOWN_PAYLOAD,
        V1[:10] + b"\x02" + V1[11:],          # version 1.2
        V2[:12] + b"\x04" + V2[13:],          # engine index 4
        V2[:11] + b"\x06" + V2[12:],          # file_type 6
        V2[:65],                              # a 1.1 header cut to 65 bytes
    ],
    ids=["major2", "unknown_payload", "v1_2", "engine4", "file_type6", "short_v2"],
)
def test_values_the_writer_never_produced_are_refused(data):
    with pytest.raises(ValueError):
        AliceFileHeader.from_bytes(data)


def rust_checkable_table():
    src = (ROOT / "tests" / "container_oracle.rs").read_text(encoding="utf-8")
    block = src[src.index("const CHECKABLE"):]
    block = block[: block.index("];")]
    rows = re.findall(r"\((None|Some\(0x([0-9A-Fa-f]{2})\)), (true|false)\)", block)
    assert rows, "the Rust table was not found"
    return [(None if p == "None" else int(h, 16), c == "true") for p, h, c in rows]


def test_the_checkable_rule_is_the_one_the_rust_reader_uses():
    rows = rust_checkable_table()
    assert len(rows) == 7
    for payload_type, checkable in rows:
        data = V1 if payload_type is None else V2[:13] + bytes([payload_type]) + V2[14:]
        header = AliceFileHeader.from_bytes(data)
        assert original_hash_checkable(header) == checkable, payload_type


def lzma_file():
    arr = np.random.default_rng(1).random(300).astype(np.float32)
    data = ALICEZip().compress(arr)
    assert AliceFileHeader.from_bytes(data).payload_type.value == 0x30
    return arr, data


def test_a_lossless_file_decompresses_and_its_hash_matches():
    arr, data = lzma_file()
    out = ALICEZip().decompress(data)
    assert np.asarray(out).tobytes() == arr.tobytes()


def test_a_wrong_original_hash_is_refused():
    _, data = lzma_file()
    at = 30  # original_hash of a version 1.1 header
    bad = data[:at] + bytes([data[at] ^ 1]) + data[at + 1:]
    with pytest.raises(ValueError, match="original_hash"):
        ALICEZip().decompress(bad)


def test_a_wrong_original_size_is_refused():
    _, data = lzma_file()
    size = struct.unpack_from("<Q", data, 14)[0]
    bad = data[:14] + struct.pack("<Q", size + 1) + data[22:]
    with pytest.raises(ValueError, match="original_size"):
        ALICEZip().decompress(bad)


def test_an_all_zero_hash_is_not_checked():
    arr, data = lzma_file()
    unrecorded = data[:30] + bytes(32) + data[62:]
    assert np.asarray(ALICEZip().decompress(unrecorded)).tobytes() == arr.tobytes()


def test_a_procedural_file_is_read_without_checking_the_hash():
    data = ALICEZip().compress(np.sin(np.linspace(0, 10, 1000)).astype(np.float32))
    assert AliceFileHeader.from_bytes(data).payload_type.value == 0x00
    other = data[:30] + bytes(b ^ 0xFF for b in data[30:62]) + data[62:]
    ALICEZip().decompress(other)


def test_a_payload_longer_than_stated_is_refused():
    _, data = lzma_file()
    # the length check must refuse it, not a later decoding failure
    with pytest.raises(ValueError, match="follow the header"):
        ALICEZip().decompress(data + b"\x00")


def test_the_writer_header_layout_is_the_reference_layout():
    # the writer's header for given fields equals the reference writer's bytes
    header = AliceFileHeader.from_bytes(V2)
    assert header.to_bytes() == V2[:66]


def test_the_writer_output_reads_back_on_this_platform():
    for v in [b"hello world, some text bytes", np.random.default_rng(1).random(300).astype(np.float32)]:
        data = ALICEZip().compress(v)
        h = AliceFileHeader.from_bytes(data)
        assert h.compressed_size == len(data) - h.header_size()
        assert h.original_hash == hashlib.sha256(np.asarray(
            np.frombuffer(v, dtype=np.uint8) if isinstance(v, bytes) else v).tobytes()).digest()


def test_the_file_path_reader_applies_the_same_checks(tmp_path):
    arr, data = lzma_file()
    good = tmp_path / "good.alice"
    good.write_bytes(data)
    assert np.asarray(ALICEZip().decompress(good)).tobytes() == arr.tobytes()
    longer = tmp_path / "longer.alice"
    longer.write_bytes(data + b"\x00")
    with pytest.raises(ValueError, match="follow the header"):
        ALICEZip().decompress(longer)
    wrong = tmp_path / "wrong.alice"
    wrong.write_bytes(data[:30] + bytes([data[30] ^ 1]) + data[31:])
    with pytest.raises(ValueError, match="original_hash"):
        ALICEZip().decompress(wrong)
    major2 = tmp_path / "major2.alice"
    major2.write_bytes(MAJOR2)
    with pytest.raises(ValueError):
        ALICEZip().decompress(major2)
