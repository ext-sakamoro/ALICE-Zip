#!/usr/bin/env python3
"""An independent implementation of the v2 law reference (`alice_zip::law_ref::LawRef`).

A law reference names one law by its kind, the semantics it is evaluated under, its law
identifier and its rows (named, typed values), and carries enough to recompute the identifier.
The bytes, every integer big-endian:

    magic "LOLR" | u16 version 2
    u8 len(kind) | kind                     1..=255 bytes of UTF-8
    semantics id (32) | law id (32)
    u8 row count                            0..=64
    each row: u8 len(name) | name           1..=255 bytes of UTF-8, the rows in strictly
                                            ascending bytewise order of their names
              u8 tag | value                tags as the v2 law identifier:
                0x01 bytes  u64 len | data      0x02 u32 (4)    0x03 u64 (8)
                0x04 f64    bits (8)            0x05 str    u64 len | UTF-8
                0x06 id (32)                    0x07 f32    bits (4)
    nothing after the last row

One law has one byte string: the rows are sorted by name (a writer sorts them, a reader refuses
rows out of order or with a repeated name). The law id must equal the v2 law identifier
(scripts/law_id_v2_reference.py) of (kind, semantics id) with, for each row in that sorted
order, the row's name as a str value and then the row's value. A reader recomputes it and refuses a reference whose id differs, so the bytes alone
say which law they name. A float row holds the value's bits; a NaN other than the canonical one
is refused, by the writer and by the reader (law_id_v2_reference.float_bits).

The probes are in tests/data/law_ref/v2_probes.txt; tests/law_ref_v2.rs reads the same file.

    python3 scripts/law_ref_v2_reference.py --check   # every expect line equals this computation
    python3 scripts/law_ref_v2_reference.py --write   # recompute the expect lines (a deliberate move)

Fails when the file holds no probe of either kind, so it never passes vacuously.
"""

from __future__ import annotations

import importlib.util
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROBES = ROOT / "tests" / "data" / "law_ref" / "v2_probes.txt"
MAGIC = b"LOLR"
VERSION = 2
MAX_ROWS = 64
TAGS = {"bytes": 0x01, "u32": 0x02, "u64": 0x03, "f64": 0x04, "str": 0x05, "id": 0x06, "f32": 0x07}


def _law_id_module():
    spec = importlib.util.spec_from_file_location(
        "law_id_v2_reference", ROOT / "scripts" / "law_id_v2_reference.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAW_ID = _law_id_module()


def law_id(kind: bytes, semantics: bytes, rows: list[tuple[str, str, object]]) -> bytes:
    values: list[tuple[str, object]] = []
    for name, t, v in rows:
        values.append(("str", name))
        values.append((t, v))
    return LAW_ID.law_id(kind, semantics, values)


def encode_value(t: str, v: object) -> bytes:
    if t == "bytes":
        return bytes([TAGS[t]]) + struct.pack(">Q", len(v)) + v
    if t == "u32":
        return bytes([TAGS[t]]) + struct.pack(">I", v)
    if t == "u64":
        return bytes([TAGS[t]]) + struct.pack(">Q", v)
    if t == "f64":
        return bytes([TAGS[t]]) + struct.pack(">Q", LAW_ID.float_bits(t, v))
    if t == "f32":
        return bytes([TAGS[t]]) + struct.pack(">I", LAW_ID.float_bits(t, v))
    if t == "str":
        b = v.encode("utf-8")
        return bytes([TAGS[t]]) + struct.pack(">Q", len(b)) + b
    if t == "id":
        return bytes([TAGS[t]]) + v
    raise ValueError(f"unknown value type {t!r}")


def sort_rows(rows: list[tuple[str, str, object]]) -> list[tuple[str, str, object]]:
    out = sorted(rows, key=lambda r: r[0].encode("utf-8"))
    if any(a[0] == b[0] for a, b in zip(out, out[1:])):
        raise ValueError("two rows have the same name")
    return out


def encode(kind: bytes, semantics: bytes, rows: list[tuple[str, str, object]]) -> bytes:
    if not 1 <= len(kind) <= 255 or len(semantics) != 32 or len(rows) > MAX_ROWS:
        raise ValueError("kind, semantics or row count out of range")
    rows = sort_rows(rows)
    out = MAGIC + struct.pack(">H", VERSION)
    out += bytes([len(kind)]) + kind + semantics + law_id(kind, semantics, rows)
    out += bytes([len(rows)])
    for name, t, v in rows:
        n = name.encode("utf-8")
        out += bytes([len(n)]) + n + encode_value(t, v)
    return out


class Refused(Exception):
    """The reason a reader refuses the bytes, named as the Rust error variant"""


def decode(b: bytes) -> tuple[bytes, bytes, bytes, list[tuple[str, str, object]]]:
    at = 0

    def take(n: int) -> bytes:
        nonlocal at
        if n > len(b) - at:
            raise Refused("Truncated")
        out = b[at : at + n]
        at += n
        return out

    def utf8(raw: bytes) -> str:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise Refused("NotUtf8") from None

    if take(4) != MAGIC:
        raise Refused("BadMagic")
    if struct.unpack(">H", take(2))[0] != VERSION:
        raise Refused("UnsupportedVersion")
    kind_len = take(1)[0]
    if kind_len == 0:
        raise Refused("EmptyKind")
    kind = take(kind_len)
    utf8(kind)
    semantics = take(32)
    stated = take(32)
    count = take(1)[0]
    if count > MAX_ROWS:
        raise Refused("TooManyRows")
    rows: list[tuple[str, str, object]] = []
    prev: bytes | None = None
    for _ in range(count):
        name_len = take(1)[0]
        if name_len == 0:
            raise Refused("EmptyName")
        raw_name = take(name_len)
        name = utf8(raw_name)
        if prev is not None and raw_name == prev:
            raise Refused("DuplicateName")
        if prev is not None and raw_name < prev:
            raise Refused("UnsortedRows")
        prev = raw_name
        tag = take(1)[0]
        if tag == 0x01:
            v: object = take(struct.unpack(">Q", take(8))[0])
            t = "bytes"
        elif tag == 0x02:
            v, t = struct.unpack(">I", take(4))[0], "u32"
        elif tag == 0x03:
            v, t = struct.unpack(">Q", take(8))[0], "u64"
        elif tag in (0x04, 0x07):
            t = "f64" if tag == 0x04 else "f32"
            raw = take(8 if tag == 0x04 else 4)
            v = int.from_bytes(raw, "big")
            try:
                LAW_ID.float_bits(t, v)
            except LAW_ID.NonCanonicalNan:
                raise Refused("NonCanonicalNan") from None
        elif tag == 0x05:
            v, t = utf8(take(struct.unpack(">Q", take(8))[0])), "str"
        elif tag == 0x06:
            v, t = take(32), "id"
        else:
            raise Refused("UnknownTag")
        rows.append((name, t, v))
    if at != len(b):
        raise Refused("TrailingBytes")
    if law_id(kind, semantics, rows) != stated:
        raise Refused("LawIdMismatch")
    return kind, semantics, stated, rows


def parse(text: str) -> list[dict]:
    """Blocks `ref <name>` (kind / semantics / row / expect_bytes / expect_id) and
    `reject <name>` (bytes / expect_error), each closed by `end`"""
    probes, cur = [], None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        word, _, rest = line.partition(" ")
        if word in ("ref", "reject"):
            cur = {"type": word, "name": rest, "rows": [], "line": n}
        elif cur is None:
            raise ValueError(f"line {n}: `{word}` outside a probe")
        elif word == "kind":
            cur["kind"] = bytes.fromhex(rest)
        elif word == "semantics":
            cur["semantics"] = bytes.fromhex(rest)
        elif word == "row":
            name, t, v = (rest.split(" ", 2) + [""])[:3]
            name = bytes.fromhex(name).decode("utf-8")
            if t in ("bytes", "id"):
                cur["rows"].append((name, t, bytes.fromhex(v)))
            elif t in ("u32", "u64"):
                cur["rows"].append((name, t, int(v)))
            elif t in ("f64", "f32"):
                cur["rows"].append((name, t, int(v, 16)))
            elif t == "str":
                cur["rows"].append((name, t, bytes.fromhex(v).decode("utf-8")))
            else:
                raise ValueError(f"line {n}: unknown value type {t!r}")
        elif word == "bytes":
            cur["bytes"] = bytes.fromhex(rest)
        elif word in ("expect_bytes", "expect_id", "expect_error"):
            cur[word] = rest
        elif word == "end":
            probes.append(cur)
            cur = None
        else:
            raise ValueError(f"line {n}: unknown line `{word}`")
    if cur is not None:
        raise ValueError("the last probe has no `end`")
    return probes


def compute(p: dict) -> dict:
    if p["type"] == "ref":
        b = encode(p["kind"], p["semantics"], p["rows"])
        decoded = decode(b)
        if decoded[3] != sort_rows(p["rows"]):
            raise ValueError(f"{p['name']}: the encoding does not decode to its rows")
        return {"expect_bytes": b.hex(), "expect_id": decoded[2].hex()}
    try:
        decode(p["bytes"])
    except Refused as e:
        return {"expect_error": str(e)}
    return {"expect_error": "accepted"}


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "--check"
    text = PROBES.read_text(encoding="utf-8")
    probes = parse(text)
    refs = sum(p["type"] == "ref" for p in probes)
    rejects = len(probes) - refs
    if refs == 0 or rejects == 0:
        print(f"law_ref_v2_reference: {refs} ref and {rejects} reject probes", file=sys.stderr)
        return 1
    got = [compute(p) for p in probes]
    if mode == "--write":
        out, i = [], 0
        for raw in text.splitlines():
            if raw.strip().split(" ", 1)[0] in ("expect_bytes", "expect_id", "expect_error"):
                continue
            if raw.strip() == "end":
                out.extend(f"{k} {v}" for k, v in got[i].items())
                i += 1
            out.append(raw)
        PROBES.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"law_ref_v2_reference: wrote {len(probes)} probes")
        return 0
    bad = [
        f"{p['name']}: {k} file {p.get(k)} computed {v}"
        for p, g in zip(probes, got)
        for k, v in g.items()
        if p.get(k) != v
    ]
    for b in bad:
        print(f"law_ref_v2_reference: {b}", file=sys.stderr)
    print(f"law_ref_v2_reference: {refs} ref + {rejects} reject probes, {len(bad)} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
