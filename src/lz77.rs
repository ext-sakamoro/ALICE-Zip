//! LZ77 sliding-window compression
//!
//! Classic LZ77 style token stream (offset / length / literal) with a
//! sliding window over previously-seen bytes.
//!
//! Law: `lz77_decode(&lz77_encode(data, w, l)) == Ok(data)` for every `data`,
//! every `w` and every `l` (the window / lookahead sizes only change the
//! token count, never the decoded bytes) — pinned by
//! `tests/analytic_oracle.rs` and the `lz77_roundtrip` fuzz target

use alloc::vec::Vec;

use crate::error::ZipError;

/// One LZ77 token: copy `length` bytes from `offset` bytes back, then emit
/// `literal` `offset == 0 && length == 0` is a pure literal
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LzToken {
    pub offset: u16,
    pub length: u16,
    pub literal: u8,
}

/// Largest window / lookahead the `u16` token fields can express
/// `lz77_encode` clamps larger requests to this value instead of truncating
pub const MAX_WINDOW: usize = u16::MAX as usize;

/// LZ77圧縮 (スライディングウィンドウ)
///
/// `window_size` / `lookahead_size` は [`MAX_WINDOW`] (65535) に clamp される
/// (token の `offset` / `length` が `u16` のため) `lookahead_size == 0` は
/// 全 byte を literal として出力する
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn lz77_encode(data: &[u8], window_size: usize, lookahead_size: usize) -> Vec<LzToken> {
    let window_size = window_size.min(MAX_WINDOW);
    let lookahead_size = lookahead_size.min(MAX_WINDOW);
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

        // Every token carries a literal. When the match would swallow the
        // last byte, shorten it by one so that byte becomes the literal
        // (versions ≤ 0.3 emitted a phantom `0` literal here, so the decoded
        // stream was one byte longer than the input)
        if pos + best_length as usize >= data.len() {
            best_length -= 1;
        }
        let literal = data[pos + best_length as usize];

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
///
/// # Errors
///
/// [`ZipError::InvalidData`] when a token references bytes before the start
/// of the output (`offset > decoded.len()`), or has `length > 0` with
/// `offset == 0` Streams produced by [`lz77_encode`] never fail
pub fn lz77_decode(tokens: &[LzToken]) -> Result<Vec<u8>, ZipError> {
    let mut result = Vec::new();
    for token in tokens {
        if token.length > 0 {
            let offset = token.offset as usize;
            if offset == 0 || offset > result.len() {
                return Err(ZipError::InvalidData);
            }
            let start = result.len() - offset;
            // The copy may overlap its own output (run-length style), so it
            // has to be byte-by-byte from the growing buffer
            for i in 0..token.length as usize {
                let byte = result[start + i];
                result.push(byte);
            }
        }
        result.push(token.literal);
    }
    Ok(result)
}

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

    #[test]
    fn lz77_roundtrip() {
        let data = b"abcabcabcabc";
        let tokens = lz77_encode(data, 256, 32);
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn lz77_no_repetition() {
        let data = b"abcdefgh";
        let tokens = lz77_encode(data, 256, 32);
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn lz77_all_same() {
        let data = alloc::vec![b'a'; 100];
        let tokens = lz77_encode(&data, 256, 32);
        assert!(tokens.len() < data.len()); // 圧縮されている
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
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
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    /// 2バイトで繰り返しなし
    #[test]
    fn lz77_two_bytes_no_match() {
        let data = b"ab";
        let tokens = lz77_encode(data, 256, 32);
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    /// 2バイト同一文字
    #[test]
    fn lz77_two_bytes_same() {
        let data = b"aa";
        let tokens = lz77_encode(data, 256, 32);
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    /// ウィンドウサイズ1で動作確認
    #[test]
    fn lz77_window_size_one() {
        let data = b"aabbaabb";
        let tokens = lz77_encode(data, 1, 32);
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    /// lookahead_size=1で動作確認
    #[test]
    fn lz77_lookahead_one() {
        let data = b"abcabcabc";
        let tokens = lz77_encode(data, 256, 1);
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    /// 全ゼロバイト列のラウンドトリップ
    #[test]
    fn lz77_all_zeros() {
        let data = alloc::vec![0u8; 200];
        let tokens = lz77_encode(&data, 256, 32);
        assert!(tokens.len() < data.len());
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    /// 全0xFFバイトのラウンドトリップ
    #[test]
    fn lz77_all_0xff() {
        let data = alloc::vec![0xFFu8; 50];
        let tokens = lz77_encode(&data, 256, 32);
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
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
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    /// 長い繰り返し文字列のラウンドトリップ
    #[test]
    fn lz77_long_repeat() {
        let data = alloc::vec![b'z'; 500];
        let tokens = lz77_encode(&data, 256, 32);
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
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
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    /// バイナリデータ（0x00〜0x0F）のラウンドトリップ
    #[test]
    fn lz77_binary_data() {
        let data: Vec<u8> = (0..16).collect();
        let tokens = lz77_encode(&data, 256, 32);
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    /// 大きなウィンドウとlookahead
    #[test]
    fn lz77_large_window() {
        let data = b"hellohellohello";
        let tokens = lz77_encode(data, 1024, 1024);
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
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
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
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
        let decoded = lz77_decode(&tokens).unwrap();
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
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, b"ababc");
    }

    /// 空トークン列のデコード
    #[test]
    fn lz77_decode_empty() {
        let tokens: Vec<LzToken> = alloc::vec![];
        let decoded = lz77_decode(&tokens).unwrap();
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
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    /// ウィンドウサイズ=0のエッジケース（参照なし）
    #[test]
    fn lz77_window_zero() {
        let data = b"abab";
        let tokens = lz77_encode(data, 0, 32);
        for t in &tokens {
            assert_eq!(t.length, 0);
        }
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
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
}
