//! The xz container (`.xz`, version 1.2.0 of the format) as Python's
//! `lzma.compress` and lzma-rs write it: one stream, one block whose only
//! filter is LZMA2, no stream padding. Every field is checked; anything else
//! is `None`. The LZMA2 chunk headers are read without decoding, so the
//! unpacked size is known before any byte is decoded.

use crc::{Crc, CRC_32_ISO_HDLC, CRC_64_XZ};

const MAGIC: [u8; 6] = [0xFD, b'7', b'z', b'X', b'Z', 0x00];
const FOOTER_MAGIC: [u8; 2] = *b"YZ";
/// Stream flags byte 1: check type none, CRC-32, CRC-64
const CHECK_NONE: u8 = 0x00;
const CHECK_CRC32: u8 = 0x01;
const CHECK_CRC64: u8 = 0x04;
const FILTER_LZMA2: u64 = 0x21;
/// Largest LZMA2 dictionary size byte accepted: 28 is 64 MiB. The
/// writer (preset 6) uses 22, 8 MiB; the format allows up to 40 (4 GiB),
/// which a decoder that allocates the dictionary would have to reserve.
const MAX_DICT_BYTE: u8 = 28;

pub(crate) fn crc32(b: &[u8]) -> u32 {
    Crc::<u32>::new(&CRC_32_ISO_HDLC).checksum(b)
}

pub(crate) fn crc64(b: &[u8]) -> u64 {
    Crc::<u64>::new(&CRC_64_XZ).checksum(b)
}

/// The check types a reader accepts
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Checks {
    /// CRC-64 only (what the Python writer emits)
    Crc64Only,
    /// None, CRC-32 or CRC-64 (lzma-rs writes none)
    NoneCrc32OrCrc64,
}

/// The integrity check of the uncompressed data a block carries
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Check {
    None,
    Crc32(u32),
    Crc64(u64),
}

impl Check {
    /// Whether `data` is what the check was computed over
    pub(crate) fn matches(self, data: &[u8]) -> bool {
        match self {
            Self::None => true,
            Self::Crc32(c) => crc32(data) == c,
            Self::Crc64(c) => crc64(data) == c,
        }
    }
}

/// The parts of a parsed stream the decoder needs.
pub(crate) struct Stream<'a> {
    /// The LZMA2 chunks of the block, up to and including the end marker.
    pub chunks: &'a [u8],
    /// Sum of the unpacked sizes the chunk headers state.
    pub unpacked: u64,
    /// The check of the uncompressed data, from the block.
    pub check: Check,
}

fn u32_le(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// Reads a multibyte integer (7 bits per byte, at most 9 bytes, no
/// redundant trailing zero byte) and returns it with its length.
fn varint(b: &[u8]) -> Option<(u64, usize)> {
    let mut v = 0_u64;
    for (i, &byte) in b.iter().enumerate().take(9) {
        v |= u64::from(byte & 0x7F) << (7 * i);
        if byte & 0x80 == 0 {
            if i > 0 && byte == 0 {
                return None;
            }
            return Some((v, i + 1));
        }
    }
    None
}

/// Walks the LZMA2 chunk headers without decoding and returns the length
/// of the chunk data (end marker included) and the unpacked total. The
/// decoder produces each chunk up to the size its header states, so the
/// total bounds what decoding the chunks can produce.
pub(crate) fn scan_lzma2(b: &[u8]) -> Option<(usize, u64)> {
    let mut at = 0_usize;
    let mut total = 0_u64;
    loop {
        let control = *b.get(at)?;
        // The first chunk resets the dictionary (uncompressed 0x01, or
        // LZMA with a dictionary reset, 0xE0 and up), as the format
        // requires; only an empty block may end at once.
        if at == 0 && !matches!(control, 0x00 | 0x01 | 0xE0..=0xFF) {
            return None;
        }
        match control {
            0x00 => return Some((at + 1, total)),
            0x01 | 0x02 => {
                let size =
                    usize::from(u16::from_be_bytes(b.get(at + 1..at + 3)?.try_into().ok()?)) + 1;
                at = at.checked_add(3 + size)?;
                total = total.checked_add(size as u64)?;
            }
            0x80..=0xFF => {
                let low = u64::from(u16::from_be_bytes(b.get(at + 1..at + 3)?.try_into().ok()?));
                let unpacked = ((u64::from(control & 0x1F) << 16) | low) + 1;
                let packed =
                    usize::from(u16::from_be_bytes(b.get(at + 3..at + 5)?.try_into().ok()?)) + 1;
                let props = usize::from(control >= 0xC0);
                // Each LZMA chunk starts a range coder, whose first byte
                // is always 0 (liblzma refuses anything else).
                if *b.get(at + 5 + props)? != 0 {
                    return None;
                }
                at = at.checked_add(5 + props + packed)?;
                total = total.checked_add(unpacked)?;
            }
            _ => return None,
        }
        if at > b.len() {
            return None;
        }
    }
}

/// Index at `at`: indicator, record count (1 with `record`, 0 without),
/// the record (unpadded size, unpacked size), padding, CRC-32. Returns
/// the index length.
fn index(b: &[u8], at: usize, record: Option<(u64, u64)>) -> Option<usize> {
    if *b.get(at)? != 0x00 {
        return None;
    }
    let mut i = at + 1;
    let (records, n) = varint(b.get(i..)?)?;
    i += n;
    match record {
        None if records == 0 => {}
        Some((unpadded, unpacked)) if records == 1 => {
            let (u, n) = varint(b.get(i..)?)?;
            i += n;
            let (v, n) = varint(b.get(i..)?)?;
            i += n;
            if u != unpadded || v != unpacked {
                return None;
            }
        }
        _ => return None,
    }
    let end = at + (i - at).next_multiple_of(4);
    if b.get(i..end)?.iter().any(|&x| x != 0) || u32_le(b, end)? != crc32(&b[at..end]) {
        return None;
    }
    Some(end + 4 - at)
}

/// Stream footer at `at`: CRC-32, backward size, flags, magic, and
/// nothing after it.
fn footer(b: &[u8], at: usize, index_size: usize, flags: [u8; 2]) -> Option<()> {
    (b.len() == at + 12
        && u32_le(b, at)? == crc32(&b[at + 4..at + 10])
        && u64::from(u32_le(b, at + 4)?) == (index_size / 4 - 1) as u64
        && b[at + 8..at + 10] == flags
        && b[at + 10..at + 12] == FOOTER_MAGIC)
        .then_some(())
}

pub(crate) fn parse(b: &[u8], checks: Checks) -> Option<Stream<'_>> {
    // Stream header: magic, flags, CRC-32 of the flags.
    if b.get(..6)? != MAGIC || u32_le(b, 8)? != crc32(b.get(6..8)?) {
        return None;
    }
    let flags: [u8; 2] = b[6..8].try_into().ok()?;
    let check_len = match (flags, checks) {
        ([0x00, CHECK_CRC64], _) => 8,
        ([0x00, CHECK_CRC32], Checks::NoneCrc32OrCrc64) => 4,
        ([0x00, CHECK_NONE], Checks::NoneCrc32OrCrc64) => 0,
        _ => return None,
    };
    // Empty input: no block, an index with no record.
    if *b.get(12)? == 0x00 {
        let index_size = index(b, 12, None)?;
        footer(b, 12 + index_size, index_size, flags)?;
        return Some(Stream {
            chunks: &[],
            unpacked: 0,
            check: Check::None,
        });
    }
    // Block header.
    let block = 12;
    let header_len = (usize::from(b[block]) + 1) * 4;
    let h = b.get(block..block + header_len)?;
    if u32_le(h, header_len - 4)? != crc32(&h[..header_len - 4]) {
        return None;
    }
    let block_flags = h[1];
    // One filter, optional sizes, reserved bits zero.
    if block_flags & 0x3C != 0 || block_flags & 0x03 != 0 {
        return None;
    }
    let mut at = 2;
    let mut stated_packed = None;
    let mut stated_unpacked = None;
    if block_flags & 0x40 != 0 {
        let (v, n) = varint(&h[at..header_len - 4])?;
        stated_packed = Some(v);
        at += n;
    }
    if block_flags & 0x80 != 0 {
        let (v, n) = varint(&h[at..header_len - 4])?;
        stated_unpacked = Some(v);
        at += n;
    }
    let (id, n) = varint(&h[at..header_len - 4])?;
    at += n;
    let (props_len, n) = varint(&h[at..header_len - 4])?;
    at += n;
    if id != FILTER_LZMA2 || props_len != 1 || *h.get(at)? > MAX_DICT_BYTE {
        return None;
    }
    at += 1;
    if at > header_len - 4 || h[at..header_len - 4].iter().any(|&x| x != 0) {
        return None;
    }

    // Compressed data, block padding, check.
    let data = block + header_len;
    let (chunks_len, unpacked) = scan_lzma2(b.get(data..)?)?;
    if stated_packed.is_some_and(|v| v != chunks_len as u64)
        || stated_unpacked.is_some_and(|v| v != unpacked)
    {
        return None;
    }
    let unpadded = header_len + chunks_len + check_len;
    let padded_end = data + chunks_len.next_multiple_of(4);
    if b.get(data + chunks_len..padded_end)?
        .iter()
        .any(|&x| x != 0)
    {
        return None;
    }
    let field = b.get(padded_end..padded_end + check_len)?;
    let check = match check_len {
        8 => Check::Crc64(u64::from_le_bytes(field.try_into().ok()?)),
        4 => Check::Crc32(u32::from_le_bytes(field.try_into().ok()?)),
        _ => Check::None,
    };

    let at = padded_end + check_len;
    let index_size = index(b, at, Some((unpadded as u64, unpacked)))?;
    footer(b, at + index_size, index_size, flags)?;
    Some(Stream {
        chunks: &b[data..data + chunks_len],
        unpacked,
        check,
    })
}
