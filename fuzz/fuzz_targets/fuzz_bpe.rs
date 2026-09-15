//! BPE: the most frequent pair occurs ≥ 2 times counting overlaps (the pair
//! count is over adjacent positions, "aaa" has (a,a) twice), replacement
//! removes exactly the non-overlapping left-to-right matches (≥ 1), entropy
//! stays within [0, 8]
//! seeds/fuzz_bpe/overlapping-pair-count: "\n\n\n" — first harness version
//! asserted ≥ 2 non-overlapping matches
#![no_main]
use alice_zip::bpe::{bpe_replace, find_most_frequent_pair};
use alice_zip::entropy::{shannon_entropy, theoretical_min_size};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 8192 {
        return;
    }
    let e = shannon_entropy(data);
    assert!((0.0..=8.0 + 1e-9).contains(&e), "{e}");
    assert!(theoretical_min_size(data) <= data.len());

    let Some(pair) = find_most_frequent_pair(data) else {
        return;
    };
    let replacement = data.first().copied().unwrap_or(0).wrapping_add(1);
    let out = bpe_replace(data, pair, replacement);
    let overlapping = data.windows(2).filter(|w| w[0] == pair.0 && w[1] == pair.1).count();
    assert!(overlapping >= 2, "{pair:?} occurs {overlapping}×");
    let mut matches = 0;
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == pair.0 && data[i + 1] == pair.1 {
            matches += 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    assert!(matches >= 1);
    assert_eq!(out.len(), data.len() - matches);
});
