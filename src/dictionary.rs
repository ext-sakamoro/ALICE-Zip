//! Dictionary coding (= phrase → index mapping with LRU-style eviction)
//!
//! A simple string-table style dictionary that assigns u32 indices to
//! variable-length byte phrases. When the entry limit is hit, the oldest
//! phrase is evicted (= arena drain + offsets shift).

use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct Dictionary {
    arena: Vec<u8>,
    offsets: Vec<u32>,
    max_entries: usize,
}

impl Dictionary {
    #[must_use]
    pub const fn new(max_entries: usize) -> Self {
        Self {
            arena: Vec::new(),
            offsets: Vec::new(),
            max_entries,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn add(&mut self, phrase: &[u8]) -> u32 {
        if let Some(pos) = (0..self.offsets.len()).position(|i| self.get_entry(i) == phrase) {
            return pos as u32;
        }
        if self.offsets.len() >= self.max_entries {
            let first_end = if self.offsets.len() > 1 {
                self.offsets[1] as usize
            } else {
                self.arena.len()
            };
            self.arena.drain(..first_end);
            self.offsets.remove(0);
            for off in &mut self.offsets {
                *off -= first_end as u32;
            }
        }
        self.offsets.push(self.arena.len() as u32);
        self.arena.extend_from_slice(phrase);
        (self.offsets.len() - 1) as u32
    }

    #[must_use]
    pub fn lookup(&self, idx: u32) -> Option<&[u8]> {
        let i = idx as usize;
        if i >= self.offsets.len() {
            return None;
        }
        Some(self.get_entry(i))
    }

    #[inline]
    fn get_entry(&self, i: usize) -> &[u8] {
        let start = self.offsets[i] as usize;
        let end = if i + 1 < self.offsets.len() {
            self.offsets[i + 1] as usize
        } else {
            self.arena.len()
        };
        &self.arena[start..end]
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.offsets.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
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
}
