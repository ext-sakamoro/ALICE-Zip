"""Decompression that refuses output longer than the caller's limit

Every reader in this package knows from a header how long a payload must be,
so it passes that length here; a stream that expands past it raises
`OutputTooLarge` after at most ``limit + 1`` bytes have been produced (the
Rust crate does the same with `DecompressError::TooLarge`).
"""

import lzma
import struct
import zlib

# the dictionary a stream may ask for: the writers (preset 6) use 8 MiB, a header can ask for 4 GiB
MAX_LZMA_DECODER_MEMORY = 80 * 1024 * 1024

# the largest block a zstd frame may hold (RFC 8878 Block_Maximum_Size)
ZSTD_BLOCK_MAX = 128 * 1024


class OutputTooLarge(ValueError):
    """The decompressed output would exceed ``limit`` bytes"""

    def __init__(self, limit: int):
        super().__init__(f"decompressed output exceeds the limit of {limit} bytes")
        self.limit = limit


class TooManyValues(ValueError):
    """A header states more values than the caller accepts"""

    def __init__(self, stated: int, limit: int):
        super().__init__(f"the data states {stated} values, more than the limit of {limit}")
        self.stated = stated
        self.limit = limit


def check_values(stated: int, limit: int) -> None:
    """TooManyValues when ``stated`` passes the caller's ``limit`` (the header is the data's own claim)"""
    _check_limit(limit)
    if stated > limit:
        raise TooManyValues(stated, limit)


def _check_limit(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError(f"limit must be an integer >= 0, got {limit!r}")


def lzma_decompress(data: bytes, limit: int, format: int = lzma.FORMAT_AUTO) -> bytes:
    """`lzma.decompress` of at most ``limit`` bytes

    Concatenated streams are read as `lzma.decompress` reads them (data after
    the first stream that is not a stream is ignored); the limit holds for
    their sum.
    """
    _check_limit(limit)
    parts = []
    total = 0
    while True:
        decoder = lzma.LZMADecompressor(format=format, memlimit=MAX_LZMA_DECODER_MEMORY)
        try:
            chunk = decoder.decompress(data, max_length=limit - total + 1)
        except lzma.LZMAError:
            if parts:
                break  # trailing data that is not a stream, as lzma.decompress ignores it
            raise
        total += len(chunk)
        if total > limit:
            raise OutputTooLarge(limit)
        parts.append(chunk)
        if not decoder.eof:
            raise lzma.LZMAError("Compressed data ended before the end-of-stream marker was reached")
        data = decoder.unused_data
        if not data:
            break
    return b"".join(parts)


def zlib_decompress(data: bytes, limit: int, wbits: int = zlib.MAX_WBITS) -> bytes:
    """`zlib.decompress` of at most ``limit`` bytes"""
    _check_limit(limit)
    decoder = zlib.decompressobj(wbits)
    out = decoder.decompress(data, limit + 1)
    if len(out) > limit:
        raise OutputTooLarge(limit)
    if not decoder.eof:
        raise zlib.error("Error -5 while decompressing data: incomplete or truncated stream")
    return out


def zstd_single_frame_size(data: bytes) -> int:
    """The content size of ``data`` when it is exactly one zstd frame (RFC 8878 3.1.1): the header must state
    the size, and the blocks (3-byte headers) and the optional checksum must end where ``data`` ends; a
    second frame or trailing bytes raise ValueError (libzstd refuses a frame whose blocks do not decode to
    the stated size)"""
    size = zstd_content_size(data)
    fhd = data[4]
    single = (fhd >> 5) & 1
    at = 5 + (0 if single else 1) + (0, 1, 2, 4)[fhd & 3] + (1 if single else 0, 2, 4, 8)[fhd >> 6]
    while True:
        if len(data) < at + 3:
            raise ValueError("zstd frame truncated")
        header = int.from_bytes(data[at:at + 3], "little")
        last, kind, block = header & 1, (header >> 1) & 3, header >> 3
        if kind == 3:
            raise ValueError("reserved zstd block type")
        # Block_Maximum_Size (RFC 8878 3.1.1.2.4): no block holds or regenerates more than 128 KiB
        if block > ZSTD_BLOCK_MAX:
            raise ValueError(f"zstd block of {block} bytes, more than 128 KiB ({ZSTD_BLOCK_MAX})")
        at += 3 + (1 if kind == 1 else block)  # an RLE block stores one byte
        if last:
            break
    at += 4 if fhd & 0x04 else 0
    if at != len(data):
        raise ValueError("bytes after the zstd frame (one frame is read)")
    return size


def zstd_content_size(data: bytes) -> int:
    """The content size a zstd frame header states (RFC 8878 3.1.1.1), or ValueError when it states none"""
    if len(data) < 5 or struct.unpack("<I", data[:4])[0] != 0xFD2FB528:
        raise ValueError("not a zstd frame")
    fhd = data[4]
    fcs_flag, single = fhd >> 6, (fhd >> 5) & 1
    pos = 5 + (0 if single else 1) + (0, 1, 2, 4)[fhd & 3]
    size = (1 if single else 0, 2, 4, 8)[fcs_flag]
    if size == 0:
        raise ValueError("the zstd frame does not state its size")
    if len(data) < pos + size:
        raise ValueError("zstd frame header truncated")
    value = int.from_bytes(data[pos:pos + size], "little")
    return value + 256 if size == 2 else value
