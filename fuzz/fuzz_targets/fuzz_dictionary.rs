//! Dictionary: arbitrary add / lookup sequences never panic, len ≤ capacity,
//! every returned index is immediately resolvable
#![no_main]
use alice_zip::dictionary::Dictionary;
use alice_zip::ZipError;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Op {
    Add(Vec<u8>),
    Lookup(u32),
}

#[derive(Arbitrary, Debug)]
struct Input {
    capacity: u8,
    ops: Vec<Op>,
}

fuzz_target!(|input: Input| {
    if input.ops.len() > 256 {
        return;
    }
    let cap = usize::from(input.capacity);
    let mut dict = Dictionary::new(cap);
    for op in &input.ops {
        match op {
            Op::Add(phrase) => match dict.add(phrase) {
                Ok(idx) => {
                    assert_eq!(dict.lookup(idx), Some(phrase.as_slice()));
                    assert!((idx as usize) < dict.len());
                }
                Err(e) => {
                    assert_eq!(e, ZipError::DictionaryFull);
                    assert_eq!(cap, 0);
                }
            },
            Op::Lookup(idx) => {
                assert_eq!(dict.lookup(*idx).is_some(), (*idx as usize) < dict.len());
            }
        }
        assert!(dict.len() <= cap);
        assert_eq!(dict.is_empty(), dict.len() == 0);
    }
});
