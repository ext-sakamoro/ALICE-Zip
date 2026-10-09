#!/usr/bin/env python3
"""Independent reference writer for the container format.

The Rust implementation in `src/container.rs` is checked against the bytes
and identifiers this script produces; the script does not call the crate.
It uses only `struct` and `hashlib`.

Run from the repository root to regenerate the fixtures and the golden
values (`python3 tests/data/container/container_ref.py`). The fixtures are
committed; the Rust tests compare against them byte for byte.

Layout (all integers little endian unless noted):

    header (56 bytes)
      [0..8)   magic 89 41 4C 43 0D 0A 1A 0A
      [8..10)  major u16 = 1
      [10..12) minor u16
      [12..16) reserved u32 = 0
      [16..48) semantics id (32 bytes)
      [48..56) section count u64
    section table: count x 56 bytes
      tag (4 bytes, stored as is) | flags u16 (bit 0 = critical)
      | reserved u16 = 0 | offset u64 | length u64 | SHA-256 of the payload
    payloads, in table order, with no gap between them
    trailer: SHA-256 of every preceding byte (32 bytes)

    container id = SHA-256( be64(len(domain)) || domain
                            || be64(56) || header
                            || be64(len(table)) || table )
    with domain = b"alice/container/v1"; the trailer is not part of the id.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

MAGIC = bytes([0x89, 0x41, 0x4C, 0x43, 0x0D, 0x0A, 0x1A, 0x0A])
DOMAIN = b"alice/container/v1"
HERE = Path(__file__).resolve().parent


def sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def be64(n: int) -> bytes:
    return struct.pack(">Q", n)


def build(semantics: bytes, sections, minor=0, major=1):
    """sections: list of (tag: bytes(4), critical: bool, payload: bytes)"""
    assert len(semantics) == 32
    header = MAGIC + struct.pack("<HHI", major, minor, 0) + semantics + struct.pack("<Q", len(sections))
    offset = len(header) + 56 * len(sections)
    table = b""
    body = b""
    for tag, critical, payload in sections:
        assert len(tag) == 4
        table += tag + struct.pack("<HHQQ", 1 if critical else 0, 0, offset, len(payload)) + sha(payload)
        offset += len(payload)
        body += payload
    data = header + table + body
    data += sha(data)
    cid = sha(be64(len(DOMAIN)) + DOMAIN + be64(len(header)) + header + be64(len(table)) + table)
    return data, cid


def sref(rows):
    """rows: list of (target_sha: 32 bytes, kind: 4 bytes, builder: bytes)"""
    out = struct.pack("<Q", len(rows))
    for target, kind, builder in rows:
        out += target + kind + struct.pack("<H", len(builder)) + builder
    return out


def lids(rows):
    """rows: list of (section_index, law_id: 32 bytes, semantics_id: 32 bytes)"""
    out = struct.pack("<Q", len(rows))
    for index, law_id, semantics in rows:
        out += struct.pack("<Q", index) + law_id + semantics
    return out


def big_payload():
    return bytes((i * 31 + 7) % 256 for i in range(1 << 20))


def legacy_alice_zip(major, minor, file_type, engine, payload_type, payload, original_size, original_hash):
    """Same struct layout as `alice_zip/core.py` `AliceFileHeader.to_bytes`
    (v2) and its v1 branch."""
    if (major, minor) >= (1, 1):
        header = struct.pack(
            "<9sBBBBB Q Q 32s I", b"ALICE_ZIP", major, minor, file_type, engine, payload_type,
            original_size, len(payload), original_hash, 0)
    else:
        header = struct.pack(
            "<9sBBBB Q Q 32s I", b"ALICE_ZIP", major, minor, file_type, engine,
            original_size, len(payload), original_hash, 0)
    return header + payload


SEM = bytes([0x11]) * 32


def scenes():
    shape = b"shape-bytes-0123456789"
    world = b"world-state"
    shape_sha = sha(shape)
    return {
        "empty": (SEM, []),
        "one": (SEM, [(b"RAW_", True, b"hello")]),
        "dup_tag": (SEM, [(b"RAW_", True, b"a"), (b"RAW_", True, b"b"), (b"PROV", False, b"note")]),
        "zero_len": (SEM, [(b"RESD", True, b"")]),
        "unknown_noncritical": (SEM, [(b"RAW_", True, b"x"), (b"ZZZZ", False, b"keep me")]),
        "bundle": (SEM, [
            (b"SHAP", True, shape),
            (b"PHYS", True, world),
            (b"SREF", True, sref([
                (shape_sha, b"SHAP", b"alice-sdf/compiled-field/v1"),
                (shape_sha, b"SHAP", b"alice-sdf/compiled-field/v1"),
            ])),
            (b"LIDS", True, lids([(1, bytes([0xAB]) * 32, SEM)])),
        ]),
    }


def main():
    golden = {}
    for name, (sem, secs) in scenes().items():
        data, cid = build(sem, secs)
        (HERE / f"{name}.bin").write_bytes(data)
        golden[name] = {"id": cid.hex(), "file_sha256": sha(data).hex(), "len": len(data)}
    data, cid = build(SEM, [(b"RAW_", True, big_payload())])
    golden["big"] = {"id": cid.hex(), "file_sha256": sha(data).hex(), "len": len(data)}

    legacy_payload = b"\x00legacy-payload\xff"
    original_hash = sha(b"original data")
    (HERE / "legacy_v1.alice").write_bytes(
        legacy_alice_zip(1, 0, 0x05, 0, 0, legacy_payload, 13, original_hash))
    (HERE / "legacy_v2.alice").write_bytes(
        legacy_alice_zip(1, 1, 0x01, 3, 0x30, legacy_payload, 13, original_hash))
    # inputs the old lenient readers accepted and the container must refuse
    (HERE / "legacy_major2.alice").write_bytes(
        legacy_alice_zip(2, 0, 0x01, 0, 0x00, legacy_payload, 13, original_hash))
    (HERE / "legacy_unknown_payload.alice").write_bytes(
        legacy_alice_zip(1, 1, 0x01, 0, 0x7F, legacy_payload, 13, original_hash))

    (HERE / "golden.json").write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(golden)} golden entries", file=sys.stderr)


if __name__ == "__main__":
    main()
