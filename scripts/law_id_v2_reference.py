#!/usr/bin/env python3
"""An independent implementation of the v2 law identifier (`alice_zip::law_id::LawIdHasher`).

The identifier of a law is SHA-256 over, in order (every length a big-endian u64):

    len(domain) domain          domain = b"alice-law-id/v2"
    len(kind)   kind            the law's kind tag, e.g. b"signal-law/polynomial/v1"
    semantics_id                32 bytes, the semantics the law is evaluated under
    then each value, a one-byte type tag first:
      0x01 bytes   len  data
      0x02 u32     4 bytes big-endian
      0x03 u64     8 bytes big-endian
      0x04 f64     8 bytes big-endian of the bits
      0x05 str     len  UTF-8 bytes
      0x06 id      32 bytes
      0x07 f32     4 bytes big-endian of the bits

A float enters by its bits, so -0.0 and +0.0 are different values and an f32 is never widened
to f64 (f32 1.5 and f64 1.5 give different identifiers). The only NaN accepted is the canonical
one, positive and quiet with an empty payload (f64 0x7ff8000000000000, f32 0x7fc00000); any
other NaN is refused (`alice_det_math::law_bits`), since a platform's default NaN differs by
target. The probe file writes floats as their bits in hex.

The tag keeps `u32 5` apart from `u64 5`, and the length prefixes keep ("ab", "c") apart from
("a", "bc"). This file uses only the standard library, so it shares no code with the Rust one.

The probes are in tests/data/law_id/v2_probes.txt; tests/law_id_v2.rs reads the same file and
must compute the same `expect` lines.

    python3 scripts/law_id_v2_reference.py --check   # every expect line equals this computation
    python3 scripts/law_id_v2_reference.py --write   # recompute the expect lines (a deliberate move)

Fails when the file holds no probe, so it never passes vacuously.
"""

from __future__ import annotations

import hashlib
import pathlib
import struct
import sys

DOMAIN = b"alice-law-id/v2"
ROOT = pathlib.Path(__file__).resolve().parent.parent
CANONICAL_NAN_F64 = 0x7FF8_0000_0000_0000
CANONICAL_NAN_F32 = 0x7FC0_0000


class NonCanonicalNan(Exception):
    """A NaN other than the canonical one"""


def float_bits(t: str, bits: int) -> int:
    if t == "f64":
        is_nan = bits & 0x7FFF_FFFF_FFFF_FFFF > 0x7FF0_0000_0000_0000
        canonical = CANONICAL_NAN_F64
    else:
        is_nan = bits & 0x7FFF_FFFF > 0x7F80_0000
        canonical = CANONICAL_NAN_F32
    if is_nan and bits != canonical:
        raise NonCanonicalNan(hex(bits))
    return bits
PROBES = ROOT / "tests" / "data" / "law_id" / "v2_probes.txt"


def be64(n: int) -> bytes:
    return struct.pack(">Q", n)


def law_id(kind: bytes, semantics: bytes, values: list[tuple[str, object]]) -> bytes:
    if len(semantics) != 32:
        raise ValueError("semantics id is not 32 bytes")
    h = hashlib.sha256()
    h.update(be64(len(DOMAIN)) + DOMAIN)
    h.update(be64(len(kind)) + kind)
    h.update(semantics)
    for t, v in values:
        if t == "bytes":
            h.update(b"\x01" + be64(len(v)) + v)
        elif t == "u32":
            h.update(b"\x02" + struct.pack(">I", v))
        elif t == "u64":
            h.update(b"\x03" + struct.pack(">Q", v))
        elif t == "f64":
            h.update(b"\x04" + struct.pack(">Q", float_bits(t, v)))
        elif t == "f32":
            h.update(b"\x07" + struct.pack(">I", float_bits(t, v)))
        elif t == "str":
            b = v.encode("utf-8")
            h.update(b"\x05" + be64(len(b)) + b)
        elif t == "id":
            if len(v) != 32:
                raise ValueError("an id value is not 32 bytes")
            h.update(b"\x06" + v)
        else:
            raise ValueError(f"unknown value type {t!r}")
    return h.digest()


def parse(text: str) -> list[dict]:
    """The probes: blocks `probe <name>` .. `end` with `kind`, `semantics`, `value`, `expect`"""
    probes, cur = [], None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        word, _, rest = line.partition(" ")
        if word == "probe":
            cur = {"name": rest, "values": [], "line": n}
        elif cur is None:
            raise ValueError(f"line {n}: `{word}` outside a probe")
        elif word == "kind":
            cur["kind"] = bytes.fromhex(rest)
        elif word == "semantics":
            cur["semantics"] = bytes.fromhex(rest)
        elif word == "value":
            t, _, v = rest.partition(" ")
            if t in ("bytes", "id"):
                cur["values"].append((t, bytes.fromhex(v)))
            elif t in ("u32", "u64"):
                cur["values"].append((t, int(v)))
            elif t in ("f64", "f32"):
                cur["values"].append((t, int(v, 16)))
            elif t == "str":
                cur["values"].append((t, bytes.fromhex(v).decode("utf-8")))
            else:
                raise ValueError(f"line {n}: unknown value type {t!r}")
        elif word == "expect":
            cur["expect"] = rest
        elif word == "end":
            probes.append(cur)
            cur = None
        else:
            raise ValueError(f"line {n}: unknown line `{word}`")
    if cur is not None:
        raise ValueError("the last probe has no `end`")
    return probes


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "--check"
    text = PROBES.read_text(encoding="utf-8")
    probes = parse(text)
    if not probes:
        print(f"law_id_v2_reference: no probe in {PROBES}", file=sys.stderr)
        return 1
    bad = []
    for p in probes:
        try:
            got = law_id(p["kind"], p["semantics"], p["values"]).hex()
        except NonCanonicalNan:
            got = "refused NonCanonicalNan"
        if mode == "--write":
            p["got"] = got
        elif p.get("expect") != got:
            bad.append(f"{p['name']}: file {p.get('expect')} computed {got}")
    if mode == "--write":
        out, i = [], 0
        for raw in text.splitlines():
            if raw.strip().startswith("expect "):
                continue
            if raw.strip() == "end":
                out.append(f"expect {probes[i]['got']}")
                i += 1
            out.append(raw)
        PROBES.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"law_id_v2_reference: wrote {len(probes)} expect lines")
        return 0
    for b in bad:
        print(f"law_id_v2_reference: {b}", file=sys.stderr)
    print(f"law_id_v2_reference: {len(probes)} probes, {len(bad)} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
