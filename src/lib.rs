//! ALICE-Zip — Compression engine
//!
//! LZ77スライディングウィンドウ、辞書符号化、エントロピー推定

#![no_std]
extern crate alloc;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// LZ77
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LzToken {
    pub offset: u16,
    pub length: u16,
    pub literal: u8,
}

/// LZ77圧縮 (スライディングウィンドウ)
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn lz77_encode(data: &[u8], window_size: usize, lookahead_size: usize) -> Vec<LzToken> {
    let mut tokens = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        let search_start = pos.saturating_sub(window_size);
        let mut best_offset = 0u16;
        let mut best_length = 0u16;

        for i in search_start..pos {
            let max_match = lookahead_size.min(data.len() - pos);
            let mut len = 0;
            while len < max_match && data[i + len] == data[pos + len] {
                len += 1;
                if i + len >= pos {
                    break;
                }
            }
            if len as u16 > best_length {
                best_length = len as u16;
                best_offset = (pos - i) as u16;
            }
        }

        let literal_pos = pos + best_length as usize;
        let literal = if literal_pos < data.len() {
            data[literal_pos]
        } else {
            0
        };

        tokens.push(LzToken {
            offset: best_offset,
            length: best_length,
            literal,
        });
        pos += best_length as usize + 1;
    }
    tokens
}

/// LZ77復元
#[must_use]
pub fn lz77_decode(tokens: &[LzToken]) -> Vec<u8> {
    let mut result = Vec::new();
    for token in tokens {
        if token.length > 0 {
            let start = result.len() - token.offset as usize;
            for i in 0..token.length as usize {
                let byte = result[start + i];
                result.push(byte);
            }
        }
        result.push(token.literal);
    }
    result
}

// ---------------------------------------------------------------------------
// Dictionary Coding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Dictionary {
    entries: Vec<Vec<u8>>,
    max_entries: usize,
}

impl Dictionary {
    #[must_use]
    pub const fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn add(&mut self, phrase: &[u8]) -> u32 {
        if let Some(pos) = self.entries.iter().position(|e| e == phrase) {
            return pos as u32;
        }
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0); // LRU evict
        }
        self.entries.push(phrase.to_vec());
        (self.entries.len() - 1) as u32
    }

    #[must_use]
    pub fn lookup(&self, idx: u32) -> Option<&[u8]> {
        self.entries.get(idx as usize).map(Vec::as_slice)
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Entropy estimation
// ---------------------------------------------------------------------------

/// シャノンエントロピー (bits per byte)
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut freq = [0u64; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let n = data.len() as f64;
    let mut entropy = 0.0;
    for &f in &freq {
        if f > 0 {
            let p = f as f64 / n;
            entropy -= p * log2_approx(p);
        }
    }
    entropy
}

#[allow(clippy::cast_precision_loss)]
fn log2_approx(x: f64) -> f64 {
    if x <= 0.0 {
        return -100.0;
    }
    let y = (x - 1.0) / (x + 1.0);
    let y2 = y * y;
    let mut sum = y;
    let mut term = y;
    for k in 1..20 {
        term *= y2;
        sum += term / f64::from(2 * k + 1);
    }
    2.0 * sum / core::f64::consts::LN_2
}

/// 理論最小圧縮サイズ (bytes)
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn theoretical_min_size(data: &[u8]) -> usize {
    let entropy = shannon_entropy(data);
    let bits = entropy * data.len() as f64;
    let v = bits / 8.0;
    let i = v as usize;
    if (i as f64) < v {
        i + 1
    } else {
        i
    }
}

// ---------------------------------------------------------------------------
// Byte-pair Encoding (BPE)
// ---------------------------------------------------------------------------

/// 最頻出バイトペアの検出
#[must_use]
pub fn find_most_frequent_pair(data: &[u8]) -> Option<(u8, u8)> {
    if data.len() < 2 {
        return None;
    }
    let mut counts = alloc::vec![0u32; 65536];
    for i in 0..data.len() - 1 {
        let key = (data[i] as usize) << 8 | data[i + 1] as usize;
        counts[key] += 1;
    }
    let max_idx = counts.iter().enumerate().max_by_key(|&(_, &c)| c)?.0;
    if counts[max_idx] < 2 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)]
    Some(((max_idx >> 8) as u8, (max_idx & 0xFF) as u8))
}

/// BPE一回の置換: (a, b) → replacement
#[must_use]
pub fn bpe_replace(data: &[u8], pair: (u8, u8), replacement: u8) -> Vec<u8> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < data.len() {
        if i + 1 < data.len() && data[i] == pair.0 && data[i + 1] == pair.1 {
            result.push(replacement);
            i += 2;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZipError {
    InvalidData,
    DecompressFailed,
    DictionaryFull,
}

impl core::fmt::Display for ZipError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidData => write!(f, "invalid data"),
            Self::DecompressFailed => write!(f, "decompress failed"),
            Self::DictionaryFull => write!(f, "dictionary full"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz77_roundtrip() {
        let data = b"abcabcabcabc";
        let tokens = lz77_encode(data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    #[test]
    fn lz77_no_repetition() {
        let data = b"abcdefgh";
        let tokens = lz77_encode(data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    #[test]
    fn lz77_all_same() {
        let data = alloc::vec![b'a'; 100];
        let tokens = lz77_encode(&data, 256, 32);
        assert!(tokens.len() < data.len()); // compressed
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    #[test]
    fn dictionary_basic() {
        let mut dict = Dictionary::new(100);
        let idx1 = dict.add(b"hello");
        let idx2 = dict.add(b"world");
        assert_eq!(dict.lookup(idx1), Some(b"hello".as_slice()));
        assert_eq!(dict.lookup(idx2), Some(b"world".as_slice()));
    }

    #[test]
    fn dictionary_dedup() {
        let mut dict = Dictionary::new(100);
        let idx1 = dict.add(b"hello");
        let idx2 = dict.add(b"hello");
        assert_eq!(idx1, idx2);
        assert_eq!(dict.len(), 1);
    }

    #[test]
    fn dictionary_eviction() {
        let mut dict = Dictionary::new(3);
        dict.add(b"a");
        dict.add(b"b");
        dict.add(b"c");
        dict.add(b"d");
        assert_eq!(dict.len(), 3);
    }

    #[test]
    fn entropy_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        let e = shannon_entropy(&data);
        assert!(e > 5.0); // uniform → high entropy
    }

    #[test]
    fn entropy_single() {
        let data = alloc::vec![42u8; 1000];
        let e = shannon_entropy(&data);
        assert!(e < 0.01);
    }

    #[test]
    fn theoretical_min() {
        let data = alloc::vec![42u8; 1000];
        let min = theoretical_min_size(&data);
        assert!(min < 10); // highly compressible
    }

    #[test]
    fn bpe_find_pair() {
        let pair = find_most_frequent_pair(b"ababab").unwrap();
        assert!(pair == (b'a', b'b') || pair == (b'b', b'a'));
    }

    #[test]
    fn bpe_replace_basic() {
        let result = bpe_replace(b"ababab", (b'a', b'b'), b'X');
        assert_eq!(result, b"XXX");
    }

    #[test]
    fn lz77_empty() {
        let tokens = lz77_encode(b"", 256, 32);
        assert!(tokens.is_empty());
    }
}
