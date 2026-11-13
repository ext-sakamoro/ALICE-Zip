"""Writes the decompression bomb fixtures of tests/bounded_decompress_memory.rs.

Each file is a small stream that expands far past any limit the tests give,
written by liblzma (Python's lzma module) so the reader is checked against
another encoder than the one it decodes with:

- `alone_zeros_64m_dict1g.lzma`: 64 MiB of zeros as an LZMA "alone" stream
  whose header declares a 1 GiB dictionary (the decoder must not reserve it)
- `xz_zeros_64m.xz`: 64 MiB of zeros as the xz stream the Python writer
  emits (one block, LZMA2, CRC-64)
- `xz_varied_crc32.xz` / `xz_varied_none.xz`: 192 KiB of varied bytes with a
  CRC-32 check and with no check (lzma-rs writes no check), which must decode;
  the tests derive from the second a stream whose first chunk header lies
  about its size (a file that does not decode would stop the public text
  check from reading it)

Run from the repository root:

    python tests/data/bomb/make_bombs.py
"""

import lzma
import pathlib
import struct

OUT = pathlib.Path(__file__).resolve().parent
MIB = 1024 * 1024


def alone_bomb():
    s = bytearray(lzma.compress(bytes(64 * MIB), format=lzma.FORMAT_ALONE))
    s[1:5] = struct.pack("<I", 1 << 30)
    return bytes(s)


def xz(data):
    return lzma.compress(data, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=6)


def varied():
    return bytes((i * 2654435761 >> 13) & 0xFF for i in range(192 * 1024))


def main():
    files = {
        "alone_zeros_64m_dict1g.lzma": alone_bomb(),
        "xz_zeros_64m.xz": xz(bytes(64 * MIB)),
        "xz_varied_crc32.xz": lzma.compress(varied(), format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC32),
        "xz_varied_none.xz": lzma.compress(varied(), format=lzma.FORMAT_XZ, check=lzma.CHECK_NONE),
    }
    for name, b in files.items():
        (OUT / name).write_bytes(b)
        print(f"{name}: {len(b)} bytes")


if __name__ == "__main__":
    main()
