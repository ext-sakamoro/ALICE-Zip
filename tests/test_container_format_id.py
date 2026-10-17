"""An independent reader of the container format identifier (FMID).

This file parses the container fixtures with its own `struct` code (it does not
import the reference writer or the crate) and checks what the format states:
section 0 of a container written with the identifier is FMID, critical, 32
bytes, equal to the SHA-256 of the format's definition, which is recomputed
here from the definition; there is no other FMID section; the containers
written before the identifier (v1_*.bin) have none. The value is also compared
with the reference writer's golden.json and with FORMAT_ID in src/container.rs.
"""
import hashlib
import json
import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "container"
MAGIC = bytes([0x89, 0x41, 0x4C, 0x43, 0x0D, 0x0A, 0x1A, 0x0A])
HEADER, ENTRY, TRAILER = 56, 56, 32
TAGS = b"RAW_ RESD LAW_ VRDT SHAP PHYS ASET SIMS PROV ORCL SREF LIDS FMID".split()
WITH_FMID = ["empty", "one", "dup_tag", "zero_len", "unknown_noncritical", "bundle"]
WITHOUT_FMID = ["v1_one", "v1_lids"]


def expected_format_id() -> bytes:
    items = [b"alice/container/format/v1", MAGIC, struct.pack("<H", 1),
             struct.pack("<Q", HEADER), struct.pack("<Q", ENTRY), struct.pack("<Q", TRAILER),
             b"alice/container/v1", b"".join(TAGS)]
    return hashlib.sha256(b"".join(struct.pack(">Q", len(x)) + x for x in items)).digest()


def sections(data: bytes):
    """[(tag, critical, payload)] after checking the layout and every hash"""
    assert data[:8] == MAGIC
    major, _minor, reserved = struct.unpack_from("<HHI", data, 8)
    assert (major, reserved) == (1, 0)
    (count,) = struct.unpack_from("<Q", data, 48)
    assert hashlib.sha256(data[:-TRAILER]).digest() == data[-TRAILER:]
    out = []
    cursor = HEADER + ENTRY * count
    for i in range(count):
        at = HEADER + ENTRY * i
        tag = data[at:at + 4]
        flags, res, offset, length = struct.unpack_from("<HHQQ", data, at + 4)
        assert res == 0 and flags in (0, 1) and offset == cursor
        payload = data[offset:offset + length]
        assert hashlib.sha256(payload).digest() == data[at + 24:at + 56]
        out.append((tag, flags == 1, payload))
        cursor += length
    assert cursor == len(data) - TRAILER
    return out


def test_the_identifier_agrees_with_the_writer_and_the_crate():
    want = expected_format_id().hex()
    golden = json.loads((DATA / "golden.json").read_text(encoding="utf-8"))
    assert golden["format_id"] == want
    src = (ROOT / "src" / "container.rs").read_text(encoding="utf-8")
    m = re.search(r'pub const FORMAT_ID: \[u8; 32\] =\s*crate::semantics::hex32\("([0-9a-f]{64})"\)', src)
    assert m, "FORMAT_ID not found in src/container.rs"
    assert m.group(1) == want


@pytest.mark.parametrize("name", WITH_FMID)
def test_section_zero_is_the_identifier(name):
    secs = sections((DATA / f"{name}.bin").read_bytes())
    assert secs, name
    tag, critical, payload = secs[0]
    assert (tag, critical, payload) == (b"FMID", True, expected_format_id())
    assert [s[0] for s in secs].count(b"FMID") == 1


@pytest.mark.parametrize("name", WITHOUT_FMID)
def test_earlier_containers_have_no_identifier(name):
    secs = sections((DATA / f"{name}.bin").read_bytes())
    assert secs, name
    assert all(s[0] != b"FMID" for s in secs)


def test_every_container_fixture_is_classified():
    on_disk = sorted(p.stem for p in DATA.glob("*.bin"))
    assert on_disk == sorted(WITH_FMID + WITHOUT_FMID)
