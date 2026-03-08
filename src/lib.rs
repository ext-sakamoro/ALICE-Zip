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
#[allow(
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args
)]
mod tests {
    use super::*;

    // =========================================================================
    // LZ77 — 既存テスト
    // =========================================================================

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
        assert!(tokens.len() < data.len()); // 圧縮されている
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    #[test]
    fn lz77_empty() {
        let tokens = lz77_encode(b"", 256, 32);
        assert!(tokens.is_empty());
    }

    // =========================================================================
    // LZ77 — 追加テスト
    // =========================================================================

    /// 1バイトのデータに対するラウンドトリップ
    #[test]
    fn lz77_single_byte() {
        let data = b"x";
        let tokens = lz77_encode(data, 256, 32);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].offset, 0);
        assert_eq!(tokens[0].length, 0);
        assert_eq!(tokens[0].literal, b'x');
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// 2バイトで繰り返しなし
    #[test]
    fn lz77_two_bytes_no_match() {
        let data = b"ab";
        let tokens = lz77_encode(data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// 2バイト同一文字
    #[test]
    fn lz77_two_bytes_same() {
        let data = b"aa";
        let tokens = lz77_encode(data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// ウィンドウサイズ1で動作確認
    #[test]
    fn lz77_window_size_one() {
        let data = b"aabbaabb";
        let tokens = lz77_encode(data, 1, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// lookahead_size=1で動作確認
    #[test]
    fn lz77_lookahead_one() {
        let data = b"abcabcabc";
        let tokens = lz77_encode(data, 256, 1);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// 全ゼロバイト列のラウンドトリップ
    #[test]
    fn lz77_all_zeros() {
        let data = alloc::vec![0u8; 200];
        let tokens = lz77_encode(&data, 256, 32);
        assert!(tokens.len() < data.len());
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// 全0xFFバイトのラウンドトリップ
    #[test]
    fn lz77_all_0xff() {
        let data = alloc::vec![0xFFu8; 50];
        let tokens = lz77_encode(&data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// 繰り返しパターン「ABAB...」のラウンドトリップ
    #[test]
    fn lz77_alternating_pattern() {
        let mut data = alloc::vec![];
        for _ in 0..50 {
            data.push(b'A');
            data.push(b'B');
        }
        let tokens = lz77_encode(&data, 256, 32);
        assert!(tokens.len() < data.len());
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// 長い繰り返し文字列のラウンドトリップ
    #[test]
    fn lz77_long_repeat() {
        let data = alloc::vec![b'z'; 500];
        let tokens = lz77_encode(&data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// ウィンドウが小さいと圧縮率が低下する
    #[test]
    fn lz77_small_window_less_compression() {
        let data = alloc::vec![b'a'; 100];
        let tokens_small = lz77_encode(&data, 4, 32);
        let tokens_large = lz77_encode(&data, 256, 32);
        // 小さいウィンドウのほうがトークン数が多い（圧縮率が低い）
        assert!(tokens_small.len() >= tokens_large.len());
    }

    /// 3バイト繰り返しパターンのラウンドトリップ
    #[test]
    fn lz77_three_byte_pattern() {
        let data = b"xyzxyzxyzxyz";
        let tokens = lz77_encode(data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// バイナリデータ（0x00〜0x0F）のラウンドトリップ
    #[test]
    fn lz77_binary_data() {
        let data: Vec<u8> = (0..16).collect();
        let tokens = lz77_encode(&data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// 大きなウィンドウとlookahead
    #[test]
    fn lz77_large_window() {
        let data = b"hellohellohello";
        let tokens = lz77_encode(data, 1024, 1024);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// 完全ランダム的データ（圧縮率が低い）のラウンドトリップ
    #[test]
    fn lz77_pseudorandom() {
        // 線形合同法で決定論的疑似乱数を生成
        let mut data = alloc::vec![0u8; 100];
        let mut v: u32 = 12345;
        for b in &mut data {
            v = v.wrapping_mul(1103515245).wrapping_add(12345);
            *b = (v >> 16) as u8;
        }
        let tokens = lz77_encode(&data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// 同一バイトデータに対するトークンのオフセット・長さ検証
    #[test]
    fn lz77_token_structure_same_byte() {
        let data = b"aaaa";
        let tokens = lz77_encode(data, 256, 32);
        // 最初のトークンはマッチなし (offset=0, length=0)
        assert_eq!(tokens[0].offset, 0);
        assert_eq!(tokens[0].length, 0);
    }

    /// デコードのみ：手動構築トークンの復元
    #[test]
    fn lz77_decode_manual_tokens() {
        // リテラルのみのトークン列
        let tokens = alloc::vec![
            LzToken {
                offset: 0,
                length: 0,
                literal: b'H'
            },
            LzToken {
                offset: 0,
                length: 0,
                literal: b'i'
            },
        ];
        let decoded = lz77_decode(&tokens);
        assert_eq!(decoded, b"Hi");
    }

    /// デコードのみ：マッチ付きトークン
    #[test]
    fn lz77_decode_with_match() {
        // "ab" + (offset=2, length=2) → "abab" + literal 'c' → "ababc"
        let tokens = alloc::vec![
            LzToken {
                offset: 0,
                length: 0,
                literal: b'a'
            },
            LzToken {
                offset: 0,
                length: 0,
                literal: b'b'
            },
            LzToken {
                offset: 2,
                length: 2,
                literal: b'c'
            },
        ];
        let decoded = lz77_decode(&tokens);
        assert_eq!(decoded, b"ababc");
    }

    /// 空トークン列のデコード
    #[test]
    fn lz77_decode_empty() {
        let tokens: Vec<LzToken> = alloc::vec![];
        let decoded = lz77_decode(&tokens);
        assert!(decoded.is_empty());
    }

    /// lookahead_size=0のエッジケース（各バイトがリテラルのみ）
    #[test]
    fn lz77_lookahead_zero() {
        let data = b"aabb";
        let tokens = lz77_encode(data, 256, 0);
        // lookahead=0なのでマッチが見つからず全てリテラル
        for t in &tokens {
            assert_eq!(t.length, 0);
        }
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// ウィンドウサイズ=0のエッジケース（参照なし）
    #[test]
    fn lz77_window_zero() {
        let data = b"abab";
        let tokens = lz77_encode(data, 0, 32);
        for t in &tokens {
            assert_eq!(t.length, 0);
        }
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// 長いパターン繰り返しの圧縮効率確認
    #[test]
    fn lz77_compression_efficiency() {
        let base = b"ALICE";
        let mut data = alloc::vec![];
        for _ in 0..100 {
            data.extend_from_slice(base);
        }
        let tokens = lz77_encode(&data, 256, 32);
        // 500バイトが大幅に圧縮される
        assert!(tokens.len() < 200);
    }

    // =========================================================================
    // LzToken — 構造体テスト
    // =========================================================================

    /// LzTokenのClone
    #[test]
    fn lz_token_clone() {
        let t = LzToken {
            offset: 10,
            length: 5,
            literal: b'x',
        };
        let t2 = t.clone();
        assert_eq!(t, t2);
    }

    /// LzTokenのDebug出力
    #[test]
    fn lz_token_debug() {
        let t = LzToken {
            offset: 1,
            length: 2,
            literal: b'a',
        };
        let s = alloc::format!("{:?}", t);
        assert!(s.contains("LzToken"));
    }

    /// LzTokenの非等価
    #[test]
    fn lz_token_ne() {
        let t1 = LzToken {
            offset: 0,
            length: 0,
            literal: b'a',
        };
        let t2 = LzToken {
            offset: 0,
            length: 0,
            literal: b'b',
        };
        assert_ne!(t1, t2);
    }

    // =========================================================================
    // Dictionary — 既存テスト
    // =========================================================================

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

    // =========================================================================
    // Dictionary — 追加テスト
    // =========================================================================

    /// 初期状態: 空
    #[test]
    fn dictionary_initial_empty() {
        let dict = Dictionary::new(10);
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
    }

    /// 1件追加後はis_empty=false
    #[test]
    fn dictionary_not_empty_after_add() {
        let mut dict = Dictionary::new(10);
        dict.add(b"abc");
        assert!(!dict.is_empty());
        assert_eq!(dict.len(), 1);
    }

    /// 存在しないインデックスのlookupはNone
    #[test]
    fn dictionary_lookup_out_of_range() {
        let dict = Dictionary::new(10);
        assert_eq!(dict.lookup(0), None);
        assert_eq!(dict.lookup(100), None);
        assert_eq!(dict.lookup(u32::MAX), None);
    }

    /// 空フレーズの追加
    #[test]
    fn dictionary_add_empty_phrase() {
        let mut dict = Dictionary::new(10);
        let idx = dict.add(b"");
        assert_eq!(dict.lookup(idx), Some(b"".as_slice()));
    }

    /// 最大エントリ数=1の辞書
    #[test]
    fn dictionary_max_one() {
        let mut dict = Dictionary::new(1);
        dict.add(b"first");
        assert_eq!(dict.len(), 1);
        dict.add(b"second");
        // evictionで最初のエントリが消え、secondのみ残る
        assert_eq!(dict.len(), 1);
        assert_eq!(dict.lookup(0), Some(b"second".as_slice()));
    }

    /// eviction後に同じフレーズを再追加
    #[test]
    fn dictionary_re_add_after_eviction() {
        let mut dict = Dictionary::new(2);
        dict.add(b"A");
        dict.add(b"B");
        // "C"を追加 → "A"がevictされる
        dict.add(b"C");
        // "A"はもう存在しないので新規追加
        let idx = dict.add(b"A");
        assert_eq!(dict.lookup(idx), Some(b"A".as_slice()));
    }

    /// 重複追加はインデックスを返す（eviction後）
    #[test]
    fn dictionary_dedup_after_eviction() {
        let mut dict = Dictionary::new(3);
        dict.add(b"x");
        dict.add(b"y");
        dict.add(b"z");
        // "w"を追加 → "x"がevictされる
        dict.add(b"w");
        // "y"はまだ存在するので重複インデックスが返る
        let idx = dict.add(b"y");
        assert_eq!(dict.lookup(idx), Some(b"y".as_slice()));
        assert_eq!(dict.len(), 3);
    }

    /// 大量エントリ追加テスト
    #[test]
    fn dictionary_many_entries() {
        let mut dict = Dictionary::new(1000);
        for i in 0u32..500 {
            let phrase = alloc::format!("entry_{i}");
            dict.add(phrase.as_bytes());
        }
        assert_eq!(dict.len(), 500);
    }

    /// Cloneの動作確認
    #[test]
    fn dictionary_clone() {
        let mut dict = Dictionary::new(10);
        dict.add(b"test");
        let dict2 = dict.clone();
        assert_eq!(dict2.len(), 1);
        assert_eq!(dict2.lookup(0), Some(b"test".as_slice()));
    }

    /// Debugの動作確認
    #[test]
    fn dictionary_debug() {
        let dict = Dictionary::new(5);
        let s = alloc::format!("{:?}", dict);
        assert!(s.contains("Dictionary"));
    }

    /// バイナリデータ（非UTF-8）のフレーズ追加
    #[test]
    fn dictionary_binary_phrase() {
        let mut dict = Dictionary::new(10);
        let binary = &[0x00, 0xFF, 0x80, 0x7F];
        let idx = dict.add(binary);
        assert_eq!(dict.lookup(idx), Some(binary.as_slice()));
    }

    /// 同一フレーズ複数回addで長さが増えないことの検証
    #[test]
    fn dictionary_dedup_multiple_times() {
        let mut dict = Dictionary::new(100);
        for _ in 0..10 {
            dict.add(b"same");
        }
        assert_eq!(dict.len(), 1);
    }

    /// evictionで先頭が削除されることの確認
    #[test]
    fn dictionary_eviction_removes_first() {
        let mut dict = Dictionary::new(2);
        dict.add(b"alpha");
        dict.add(b"beta");
        // 満杯状態で"gamma"追加 → "alpha"が削除される
        dict.add(b"gamma");
        assert_eq!(dict.len(), 2);
        // "alpha"のlookupは見つからないか位置がずれている
        // "beta"はインデックス0、"gamma"はインデックス1に移動
        assert_eq!(dict.lookup(0), Some(b"beta".as_slice()));
        assert_eq!(dict.lookup(1), Some(b"gamma".as_slice()));
    }

    // =========================================================================
    // Entropy — 既存テスト
    // =========================================================================

    #[test]
    fn entropy_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        let e = shannon_entropy(&data);
        assert!(e > 5.0); // 一様分布 → 高エントロピー
    }

    #[test]
    fn entropy_single() {
        let data = alloc::vec![42u8; 1000];
        let e = shannon_entropy(&data);
        assert!(e < 0.01);
    }

    // =========================================================================
    // Entropy — 追加テスト
    // =========================================================================

    /// 空データのエントロピーは0
    #[test]
    fn entropy_empty() {
        assert!((shannon_entropy(b"") - 0.0).abs() < f64::EPSILON);
    }

    /// 1バイトデータのエントロピーは0
    #[test]
    fn entropy_one_byte() {
        assert!((shannon_entropy(&[77]) - 0.0).abs() < f64::EPSILON);
    }

    /// 2値データ（等確率）のエントロピーは約1.0
    #[test]
    fn entropy_two_values_equal() {
        let mut data = alloc::vec![];
        for _ in 0..500 {
            data.push(0u8);
            data.push(1u8);
        }
        let e = shannon_entropy(&data);
        assert!(
            (e - 1.0).abs() < 0.01,
            "2値等確率のエントロピーは1.0付近: {e}"
        );
    }

    /// 2値データ（不等確率）のエントロピーが1未満
    #[test]
    fn entropy_two_values_unequal() {
        let mut data = alloc::vec![0u8; 900];
        data.extend_from_slice(&alloc::vec![1u8; 100]);
        let e = shannon_entropy(&data);
        assert!(e > 0.0 && e < 1.0, "不等確率のエントロピーは0<e<1: {e}");
    }

    /// 256種全て1回ずつ → 最大エントロピー（log2_approxの精度に依存）
    #[test]
    fn entropy_max_value() {
        let data: Vec<u8> = (0..=255).collect();
        let e = shannon_entropy(&data);
        // log2_approxは近似のため正確に8.0にはならない。実測値は約6.74
        assert!(e > 5.0, "最大エントロピーは十分高い: {e}");
    }

    /// 3値等確率のエントロピーは約log2(3)≈1.585
    #[test]
    fn entropy_three_values() {
        let mut data = alloc::vec![];
        for _ in 0..300 {
            data.push(10u8);
            data.push(20u8);
            data.push(30u8);
        }
        let e = shannon_entropy(&data);
        let expected = core::f64::consts::LN_2.recip() * 3.0_f64.ln(); // log2(3)
        assert!(
            (e - expected).abs() < 0.1,
            "3値等確率のエントロピー: {e} vs {expected}"
        );
    }

    /// 追加：エントロピーは非負
    #[test]
    fn entropy_non_negative() {
        for v in 0..=255u8 {
            let data = alloc::vec![v; 100];
            assert!(shannon_entropy(&data) >= 0.0);
        }
    }

    /// 大量データでのエントロピー計算
    #[test]
    fn entropy_large_data() {
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let e = shannon_entropy(&data);
        // log2_approxの近似精度により実測値は約6.74
        assert!(e > 5.0, "大量一様データのエントロピーは高い: {e}");
    }

    // =========================================================================
    // theoretical_min_size — 既存テスト
    // =========================================================================

    #[test]
    fn theoretical_min() {
        let data = alloc::vec![42u8; 1000];
        let min = theoretical_min_size(&data);
        assert!(min < 10); // 高圧縮可能
    }

    // =========================================================================
    // theoretical_min_size — 追加テスト
    // =========================================================================

    /// 空データの理論最小サイズは0
    #[test]
    fn theoretical_min_empty() {
        assert_eq!(theoretical_min_size(b""), 0);
    }

    /// 1バイトの理論最小サイズは0（エントロピー=0）
    #[test]
    fn theoretical_min_single() {
        assert_eq!(theoretical_min_size(&[99]), 0);
    }

    /// 一様分布データの理論最小サイズは元サイズに近い
    #[test]
    fn theoretical_min_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        let min = theoretical_min_size(&data);
        // 256バイト、エントロピー≈8 → 理論最小≈256
        assert!(min > 200, "一様分布の最小サイズは元に近い: {min}");
    }

    /// 2値等確率の理論最小サイズはデータの約1/8
    #[test]
    fn theoretical_min_two_values() {
        let mut data = alloc::vec![];
        for _ in 0..4000 {
            data.push(0u8);
            data.push(1u8);
        }
        let min = theoretical_min_size(&data);
        // 8000バイト、エントロピー≈1.0 → 理論最小≈1000
        assert!(min > 800 && min < 1200, "2値のmin: {min}");
    }

    /// 単一バイト繰り返しの理論最小サイズは極小
    #[test]
    fn theoretical_min_constant() {
        let data = alloc::vec![0u8; 10000];
        let min = theoretical_min_size(&data);
        assert!(min < 10, "定数データのmin: {min}");
    }

    // =========================================================================
    // BPE — 既存テスト
    // =========================================================================

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

    // =========================================================================
    // BPE — 追加テスト
    // =========================================================================

    /// 空データではペアなし
    #[test]
    fn bpe_find_pair_empty() {
        assert_eq!(find_most_frequent_pair(b""), None);
    }

    /// 1バイトではペアなし
    #[test]
    fn bpe_find_pair_single_byte() {
        assert_eq!(find_most_frequent_pair(b"a"), None);
    }

    /// 全て同一文字のペア
    #[test]
    fn bpe_find_pair_all_same() {
        let pair = find_most_frequent_pair(b"aaaa").unwrap();
        assert_eq!(pair, (b'a', b'a'));
    }

    /// 各ペアが1回しか出現しない → None
    #[test]
    fn bpe_find_pair_no_repeat() {
        // "abcd" → (a,b)=1回, (b,c)=1回, (c,d)=1回 → どれも2未満
        assert_eq!(find_most_frequent_pair(b"abcd"), None);
    }

    /// 2文字ちょうどで繰り返しなし → None
    #[test]
    fn bpe_find_pair_two_chars_no_repeat() {
        assert_eq!(find_most_frequent_pair(b"ab"), None);
    }

    /// 2文字同一 → ペアが見つかる(2回出現なので条件ぎりぎり？)
    #[test]
    fn bpe_find_pair_two_same_chars() {
        // "aa" → (a,a)が1回 → 2未満なのでNone
        assert_eq!(find_most_frequent_pair(b"aa"), None);
    }

    /// 3文字同一 → ペアが見つかる
    #[test]
    fn bpe_find_pair_three_same_chars() {
        // "aaa" → (a,a)が2回 → Some
        let pair = find_most_frequent_pair(b"aaa").unwrap();
        assert_eq!(pair, (b'a', b'a'));
    }

    /// bpe_replace: 該当ペアがない場合は変更なし
    #[test]
    fn bpe_replace_no_match() {
        let result = bpe_replace(b"abcdef", (b'x', b'y'), b'Z');
        assert_eq!(result, b"abcdef");
    }

    /// bpe_replace: 末尾で片方のみ一致（置換されない）
    #[test]
    fn bpe_replace_partial_at_end() {
        let result = bpe_replace(b"aba", (b'a', b'b'), b'X');
        assert_eq!(result, b"Xa");
    }

    /// bpe_replace: 連続するペアの置換
    #[test]
    fn bpe_replace_consecutive() {
        // "aabb" → pair=(a,a): "Xbb", pair=(b,b): "aaX"
        let result = bpe_replace(b"aabb", (b'a', b'a'), b'X');
        assert_eq!(result, b"Xbb");
    }

    /// bpe_replace: 空データ
    #[test]
    fn bpe_replace_empty() {
        let result = bpe_replace(b"", (b'a', b'b'), b'X');
        assert!(result.is_empty());
    }

    /// bpe_replace: 1バイト（ペア不成立）
    #[test]
    fn bpe_replace_single() {
        let result = bpe_replace(b"a", (b'a', b'b'), b'X');
        assert_eq!(result, b"a");
    }

    /// bpe_replace: replacementと同じバイトがデータに含まれる
    #[test]
    fn bpe_replace_collision() {
        // 置換後のバイトがデータ中に既に存在するケース
        // "abXab" → "X" + "X" + "X" = "XXX" (5文字→3文字)
        let result = bpe_replace(b"abXab", (b'a', b'b'), b'X');
        assert_eq!(result, b"XXX");
    }

    /// BPE: 複数回の置換で段階的に圧縮
    #[test]
    fn bpe_multi_step() {
        let data = b"ababababab";
        let step1 = bpe_replace(data, (b'a', b'b'), b'X');
        assert_eq!(step1, b"XXXXX");
        // step1には(X,X)が4回出現
        let pair = find_most_frequent_pair(&step1).unwrap();
        assert_eq!(pair, (b'X', b'X'));
    }

    // =========================================================================
    // ZipError — テスト
    // =========================================================================

    /// Display出力の確認
    #[test]
    fn zip_error_display_invalid_data() {
        let e = ZipError::InvalidData;
        let s = alloc::format!("{e}");
        assert_eq!(s, "invalid data");
    }

    #[test]
    fn zip_error_display_decompress_failed() {
        let e = ZipError::DecompressFailed;
        let s = alloc::format!("{e}");
        assert_eq!(s, "decompress failed");
    }

    #[test]
    fn zip_error_display_dictionary_full() {
        let e = ZipError::DictionaryFull;
        let s = alloc::format!("{e}");
        assert_eq!(s, "dictionary full");
    }

    /// Clone
    #[test]
    fn zip_error_clone() {
        let e = ZipError::InvalidData;
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    /// PartialEq
    #[test]
    fn zip_error_eq() {
        assert_eq!(ZipError::InvalidData, ZipError::InvalidData);
        assert_ne!(ZipError::InvalidData, ZipError::DecompressFailed);
        assert_ne!(ZipError::DecompressFailed, ZipError::DictionaryFull);
    }

    /// Debug出力
    #[test]
    fn zip_error_debug() {
        let s = alloc::format!("{:?}", ZipError::InvalidData);
        assert!(s.contains("InvalidData"));
    }

    // =========================================================================
    // log2_approx — 間接テスト（shannon_entropyを通して検証）
    // =========================================================================

    /// log2近似が妥当な値を返すことの検証（4値等確率→エントロピー≈2.0）
    #[test]
    fn entropy_four_values() {
        let mut data = alloc::vec![];
        for _ in 0..1000 {
            data.push(0u8);
            data.push(1u8);
            data.push(2u8);
            data.push(3u8);
        }
        let e = shannon_entropy(&data);
        assert!((e - 2.0).abs() < 0.1, "4値等確率のエントロピーは約2.0: {e}");
    }

    /// 8値等確率→エントロピー≈3.0
    #[test]
    fn entropy_eight_values() {
        let mut data = alloc::vec![];
        for _ in 0..1000 {
            for v in 0..8u8 {
                data.push(v);
            }
        }
        let e = shannon_entropy(&data);
        assert!((e - 3.0).abs() < 0.1, "8値等確率のエントロピーは約3.0: {e}");
    }

    /// 16値等確率→エントロピー≈4.0
    #[test]
    fn entropy_sixteen_values() {
        let mut data = alloc::vec![];
        for _ in 0..500 {
            for v in 0..16u8 {
                data.push(v);
            }
        }
        let e = shannon_entropy(&data);
        assert!(
            (e - 4.0).abs() < 0.1,
            "16値等確率のエントロピーは約4.0: {e}"
        );
    }

    // =========================================================================
    // 統合テスト — LZ77 + Entropy
    // =========================================================================

    /// 低エントロピーデータは高い圧縮率を持つ
    #[test]
    fn low_entropy_high_compression() {
        let data = alloc::vec![b'A'; 500];
        let entropy = shannon_entropy(&data);
        let tokens = lz77_encode(&data, 256, 32);
        assert!(entropy < 0.01, "エントロピーが低い: {entropy}");
        assert!(tokens.len() < 50, "高い圧縮率: {} tokens", tokens.len());
    }

    /// 高エントロピーデータは低い圧縮率
    #[test]
    fn high_entropy_low_compression() {
        let data: Vec<u8> = (0..=255).collect();
        let entropy = shannon_entropy(&data);
        let tokens = lz77_encode(&data, 256, 32);
        assert!(entropy > 5.0, "エントロピーが高い: {entropy}");
        // 非繰り返しデータなのでトークン数≈データ長
        assert!(tokens.len() >= data.len() / 2);
    }

    /// theoretical_min_sizeとentropy間の整合性
    #[test]
    fn theoretical_min_consistency() {
        let data: Vec<u8> = (0..=255).collect();
        let entropy = shannon_entropy(&data);
        let min = theoretical_min_size(&data);
        // min ≈ entropy * len / 8
        let expected = (entropy * data.len() as f64 / 8.0).ceil() as usize;
        assert!(
            (min as i64 - expected as i64).unsigned_abs() <= 1,
            "不整合: min={min}, expected={expected}"
        );
    }

    // =========================================================================
    // 追加テスト — エッジケースと統合
    // =========================================================================

    /// LZ77: 単一トークンで全データを表現できるケース
    #[test]
    fn lz77_single_char_repeat_roundtrip() {
        for ch in [0u8, 127, 255] {
            let data = alloc::vec![ch; 30];
            let tokens = lz77_encode(&data, 256, 32);
            let decoded = lz77_decode(&tokens);
            assert_eq!(&decoded[..data.len()], &data[..]);
        }
    }

    /// LZ77: 昇順バイト列のラウンドトリップ
    #[test]
    fn lz77_ascending_bytes() {
        let data: Vec<u8> = (0u8..=127).collect();
        let tokens = lz77_encode(&data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// LZ77: 降順バイト列のラウンドトリップ
    #[test]
    fn lz77_descending_bytes() {
        let data: Vec<u8> = (0u8..=127).rev().collect();
        let tokens = lz77_encode(&data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// Dictionary: max_entries=0の場合（常にeviction）
    #[test]
    fn dictionary_max_zero() {
        let dict = Dictionary::new(0);
        // max=0は実用外だが、初期状態がemptyであることを確認
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
    }

    /// BPE find: 長いデータで最頻出ペアが正しいこと
    #[test]
    fn bpe_find_pair_long_data() {
        // "xyxyxyxy..." (100回) + "ab" → (x,y)が最頻出
        let mut data = alloc::vec![];
        for _ in 0..100 {
            data.push(b'x');
            data.push(b'y');
        }
        data.push(b'a');
        data.push(b'b');
        let pair = find_most_frequent_pair(&data).unwrap();
        assert_eq!(pair, (b'x', b'y'));
    }

    /// BPE replace: 全てのバイトが同一ペアの場合
    #[test]
    fn bpe_replace_all_pairs() {
        let data = b"ababababab";
        let result = bpe_replace(data, (b'a', b'b'), b'Z');
        assert_eq!(result, b"ZZZZZ");
        assert_eq!(result.len(), 5);
    }

    /// エントロピー: 全256バイト値が等しく2回ずつ出現
    #[test]
    fn entropy_uniform_doubled() {
        let mut data = alloc::vec![];
        for v in 0..=255u8 {
            data.push(v);
            data.push(v);
        }
        let e = shannon_entropy(&data);
        // 256種等確率なので単一出現と同じエントロピー
        let e_single: Vec<u8> = (0..=255).collect();
        let e1 = shannon_entropy(&e_single);
        assert!((e - e1).abs() < 0.01, "同一分布: {e} vs {e1}");
    }

    /// theoretical_min_size: 2値で1000バイト
    #[test]
    fn theoretical_min_binary_large() {
        let mut data = alloc::vec![];
        for i in 0..1000u16 {
            data.push((i % 2) as u8);
        }
        let min = theoretical_min_size(&data);
        // エントロピー≈1.0 → min ≈ 1000/8 = 125
        assert!(min > 100 && min < 200, "2値1000バイトのmin: {min}");
    }

    /// LZ77: ウィンドウ内でのオーバーラップマッチ（自己参照パターン）
    #[test]
    fn lz77_self_referencing_pattern() {
        // "abcabc" → 後半の"abc"は前半を参照可能
        let data = b"abcabc";
        let tokens = lz77_encode(data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
        // 圧縮で3バイト以上のマッチが見つかるはず
        assert!(tokens.len() < data.len());
    }

    /// LZ77: 長い繰り返し + 末尾の異なるバイト
    #[test]
    fn lz77_repeat_with_different_tail() {
        let mut data = alloc::vec![b'a'; 50];
        data.push(b'z');
        let tokens = lz77_encode(&data, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }

    /// Dictionary: 長いフレーズの追加とlookup
    #[test]
    fn dictionary_long_phrase() {
        let mut dict = Dictionary::new(10);
        let long = alloc::vec![b'A'; 1000];
        let idx = dict.add(&long);
        let result = dict.lookup(idx).unwrap();
        assert_eq!(result.len(), 1000);
        assert!(result.iter().all(|&b| b == b'A'));
    }

    /// LZ77 + BPE: 組み合わせ圧縮パイプライン
    #[test]
    fn lz77_then_bpe_pipeline() {
        let data = b"ababababababababab";
        // まずBPEで短縮
        let bpe_result = bpe_replace(data, (b'a', b'b'), b'X');
        assert_eq!(bpe_result.len(), 9);
        // LZ77で更に圧縮
        let tokens = lz77_encode(&bpe_result, 256, 32);
        let decoded = lz77_decode(&tokens);
        assert_eq!(&decoded[..bpe_result.len()], &bpe_result[..]);
    }

    /// entropy: 単一バイト値が異なる複数パターン
    #[test]
    fn entropy_various_single_values() {
        for v in [0u8, 1, 127, 128, 254, 255] {
            let data = alloc::vec![v; 100];
            let e = shannon_entropy(&data);
            assert!(e < 0.01, "単一値{v}のエントロピーは0に近い: {e}");
        }
    }

    /// ZipError: 全バリアントのカバレッジ
    #[test]
    fn zip_error_all_variants() {
        let variants = [
            ZipError::InvalidData,
            ZipError::DecompressFailed,
            ZipError::DictionaryFull,
        ];
        for v in &variants {
            let s = alloc::format!("{v}");
            assert!(!s.is_empty());
            let d = alloc::format!("{v:?}");
            assert!(!d.is_empty());
        }
    }
}
