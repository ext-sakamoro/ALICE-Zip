//! Cross-module integration tests (= 複数 module を跨ぐ性質の test)
//!
//! LZ77 と Entropy の整合性、Dictionary + BPE の統合等、単一 module 内の
//! test では表現できない性質を検証する

#![allow(
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args
)]

use alloc::vec::Vec;

use crate::bpe::{bpe_replace, find_most_frequent_pair};
use crate::dictionary::Dictionary;
use crate::entropy::{shannon_entropy, theoretical_min_size};
use crate::error::ZipError;
use crate::lz77::{lz77_decode, lz77_encode};

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
        let decoded = lz77_decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }
}

/// LZ77: 昇順バイト列のラウンドトリップ
#[test]
fn lz77_ascending_bytes() {
    let data: Vec<u8> = (0u8..=127).collect();
    let tokens = lz77_encode(&data, 256, 32);
    let decoded = lz77_decode(&tokens).unwrap();
    assert_eq!(decoded, data);
}

/// LZ77: 降順バイト列のラウンドトリップ
#[test]
fn lz77_descending_bytes() {
    let data: Vec<u8> = (0u8..=127).rev().collect();
    let tokens = lz77_encode(&data, 256, 32);
    let decoded = lz77_decode(&tokens).unwrap();
    assert_eq!(decoded, data);
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
    let decoded = lz77_decode(&tokens).unwrap();
    assert_eq!(decoded, data);
    // 圧縮で3バイト以上のマッチが見つかるはず
    assert!(tokens.len() < data.len());
}

/// LZ77: 長い繰り返し + 末尾の異なるバイト
#[test]
fn lz77_repeat_with_different_tail() {
    let mut data = alloc::vec![b'a'; 50];
    data.push(b'z');
    let tokens = lz77_encode(&data, 256, 32);
    let decoded = lz77_decode(&tokens).unwrap();
    assert_eq!(decoded, data);
}

/// Dictionary: 長いフレーズの追加とlookup
#[test]
fn dictionary_long_phrase() {
    let mut dict = Dictionary::new(10);
    let long = alloc::vec![b'A'; 1000];
    let idx = dict.add(&long).unwrap();
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
    let decoded = lz77_decode(&tokens).unwrap();
    assert_eq!(decoded, bpe_result);
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
