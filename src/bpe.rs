//! Byte-pair Encoding (BPE) primitives
//!
//! Finds the most frequent adjacent byte pair and rewrites all its
//! occurrences with a single replacement byte a building block for
//! iterative BPE tokenizers.

use alloc::vec::Vec;

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
}
