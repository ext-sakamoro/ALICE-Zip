#!/usr/bin/env python3
"""Write the PKZIP archives of tests/pkzip_bounded.rs with Python's zipfile.

The archives come from a ZIP writer independent of `alice_zip::pkzip`: this
script, through the standard library. They vary what the reader has to handle:
stored and deflated entries, directory entries and an archive comment, an extra
field that pads entry data (as torch's writer does), a data descriptor after
the entry data (bit 3, written when the output cannot seek), ZIP64 extra fields
in the local headers, and ZIP64 records in the central directory and the end
of central directory (zipfile writes those once its limits are crossed; the
limits are lowered here so a small archive crosses them).

Entry contents follow `content(index, length)`: byte i of the entry with index k
is `(31 * i + 7 * k) % 251`. The Rust test recomputes the same formula, so the
expected bytes do not come from the reader under test. `expected.tsv` lists,
per archive and entry, the name, the method, the length and the CRC-32 that
this script computed.

    python3 scripts/gen_pkzip_fixtures.py            write the archives
    python3 scripts/gen_pkzip_fixtures.py --check    fail if expected.tsv differs

The deflated bytes depend on the zlib build, so --check compares expected.tsv
(names, methods, lengths and CRC-32 of the contents) and that every archive
still opens with zipfile and yields those contents, not the archive bytes.
"""
from __future__ import annotations

import io
import sys
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "pkzip"
DATE = (2020, 1, 1, 0, 0, 0)


def content(index: int, length: int) -> bytes:
    return bytes((31 * i + 7 * index) % 251 for i in range(length))


class NoSeek(io.RawIOBase):
    """A write-only stream that cannot seek, so zipfile writes data descriptors"""

    def __init__(self) -> None:
        self.buf = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:  # type: ignore[override]
        self.buf += b
        return len(b)


def info(name: str, method: int, extra: bytes = b"") -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, date_time=DATE)
    zi.compress_type = method
    zi.external_attr = 0o644 << 16
    zi.extra = extra
    return zi


def write(zf: zipfile.ZipFile, entries: list[tuple[str, int, int, bytes]], **kw) -> list[tuple[str, int, int, int]]:
    rows = []
    for k, (name, method, length, extra) in enumerate(entries):
        data = b"" if name.endswith("/") else content(k, length)
        with zf.open(info(name, method, extra), "w", **kw) as w:
            w.write(data)
        rows.append((name, method, len(data), zlib.crc32(data)))
    return rows


S, D = zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED


def build() -> dict[str, tuple[bytes, list[tuple[str, int, int, int]]]]:
    out: dict[str, tuple[bytes, list[tuple[str, int, int, int]]]] = {}

    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as zf:
        rows = write(zf, [("a.bin", S, 1000, b""), ("b.bin", D, 50000, b""), ("dir/", S, 0, b""),
                          ("dir/c.txt", D, 1, b""), ("empty.bin", S, 0, b""), ("empty.z", D, 0, b"")])
        zf.comment = b"an archive comment"
    out["basic.zip"] = (b.getvalue(), rows)

    # an extra field (id 0x4650, as torch writes) that pads each entry's data to 64 bytes; the local and
    # central extra fields then differ in length, so the data offset has to come from the local header
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as zf:
        rows = []
        for k, (name, length) in enumerate([("archive/data.pkl", 300), ("archive/data/0", 40), ("archive/data/1", 7)]):
            # data starts at offset + 30 + len(name) + 4 + pad
            pad = (-(b.tell() + 30 + len(name) + 4)) % 64
            data = content(k, length)
            with zf.open(info(name, S, b"PF" + pad.to_bytes(2, "little") + b"Z" * pad), "w") as w:
                w.write(data)
            rows.append((name, S, length, zlib.crc32(data)))
    out["aligned.zip"] = (b.getvalue(), rows)

    s = NoSeek()
    with zipfile.ZipFile(s, "w") as zf:
        rows = write(zf, [("x.bin", D, 4000, b""), ("y.bin", S, 33, b""), ("z.bin", D, 5, b"")])
    out["descriptor.zip"] = (bytes(s.buf), rows)

    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as zf:
        rows = write(zf, [("p.bin", S, 70, b""), ("q.bin", D, 9000, b"")], force_zip64=True)
    out["zip64_local.zip"] = (b.getvalue(), rows)

    # zipfile writes ZIP64 extra fields in the central directory and the ZIP64 end records once an offset or the
    # entry count crosses its limits: lowering them makes a small archive carry every ZIP64 record
    saved = (zipfile.ZIP64_LIMIT, zipfile.ZIP_FILECOUNT_LIMIT)
    zipfile.ZIP64_LIMIT, zipfile.ZIP_FILECOUNT_LIMIT = 64, 2
    try:
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as zf:
            rows = write(zf, [("m0.bin", S, 100, b""), ("m1.bin", D, 3000, b""), ("m2.bin", S, 80, b"")], force_zip64=True)
        out["zip64_all.zip"] = (b.getvalue(), rows)
    finally:
        zipfile.ZIP64_LIMIT, zipfile.ZIP_FILECOUNT_LIMIT = saved
    return out


def tsv(built) -> str:
    lines = ["# archive\tname\tmethod\tlength\tcrc32 (written by scripts/gen_pkzip_fixtures.py)"]
    for archive, (_, rows) in built.items():
        for name, method, length, crc in rows:
            lines.append(f"{archive}\t{name}\t{method}\t{length}\t{crc:08x}")
    return "\n".join(lines) + "\n"


def readable(path: Path, rows) -> bool:
    with zipfile.ZipFile(path) as zf:
        for k, (name, _, length, crc) in enumerate(rows):
            data = zf.read(name)
            if len(data) != length or zlib.crc32(data) != crc:
                return False
    return True


def main(argv: list[str]) -> int:
    built = build()
    text = tsv(built)
    if "--check" in argv:
        current = (OUT / "expected.tsv").read_text(encoding="utf-8")
        ok = current == text and all(readable(OUT / a, rows) for a, (_, rows) in built.items())
        print(f"gen_pkzip_fixtures: {len(built)} archive(s) compared, {'ok' if ok else 'DIFFERS'}")
        return 0 if ok and built else 1
    OUT.mkdir(parents=True, exist_ok=True)
    for archive, (data, _) in built.items():
        (OUT / archive).write_bytes(data)
    (OUT / "expected.tsv").write_text(text, encoding="utf-8", newline="\n")
    print(f"gen_pkzip_fixtures: wrote {len(built)} archive(s) to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
