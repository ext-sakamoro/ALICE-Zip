//! The bytes the encoders give, pinned: [`compression::DEFLATE_BYTES_ID`] /
//! [`compression::LZMA_BYTES_ID`] and the per-output lengths and digests behind them.
//!
//! Two kinds of expectation, kept apart:
//!
//! - **independent** — the probe inputs and the fold are checked against
//!   `tests/data/codec_bytes/reference.txt`, which `scripts/codec_bytes_reference.py` writes
//!   from the definitions (the CI runs it with `--check`)
//! - **recorded** — the compressed bytes themselves are a change detector: an encoder has no
//!   unique correct output, so the table below is this crate's encoder as resolved by its
//!   lock. Each output is also decoded (here, and by Python's zlib / lzma through
//!   `ALICE_ZIP_DUMP_CODEC_OUTPUTS` and the reference script) and must give its input back.
//!
//! Re-recording a digest is deliberate: it means compressed sizes counted by callers move.

#![cfg(feature = "std")]

use alice_zip::compression::{
    check_codec_bytes, codec_probe_inputs, deflate_bytes_id, deflate_probe_outputs,
    fold_codec_outputs, zlib_decompress, DEFLATE_BYTES_ID,
};
use sha2::{Digest, Sha256};
use std::io::Read;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn reference() -> Vec<String> {
    let text = include_str!("data/codec_bytes/reference.txt");
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

type Outputs = Vec<(String, Vec<u8>)>;

/// Recorded: `(label, length, SHA-256 of the compressed bytes)` in fold order
const DEFLATE_OUTPUTS: &[(&str, usize, &str)] = &[
    (
        "zlib/0/empty",
        11,
        "b8f45bcbd81d3edbe2fe2eaddfe03209b3dd10924cb291d4b79a47d63bb8cef9",
    ),
    (
        "deflate/0/empty",
        5,
        "f067985d352d2da6dfaef4844a66d06c5371ecbd9530a4d195ac599fef8b3427",
    ),
    (
        "zlib/1/empty",
        8,
        "09d469dfeeaf4c436fd3f80ba7e168bcc34e5050993b5da3c8416fbae680f18f",
    ),
    (
        "deflate/1/empty",
        2,
        "9b4fb24edd6d1d8830e272398263cdbf026b97392cc35387b991dc0248a628f9",
    ),
    (
        "zlib/6/empty",
        8,
        "aca503def43dc086a5aaa5c3d6f86bcbbc2aa6a80007ddea4a81bce2b376fceb",
    ),
    (
        "deflate/6/empty",
        2,
        "9b4fb24edd6d1d8830e272398263cdbf026b97392cc35387b991dc0248a628f9",
    ),
    (
        "zlib/9/empty",
        8,
        "b171e283c6145acf2b923098dbbc40ffc39b4f1db0212928f9869747376c4ac8",
    ),
    (
        "deflate/9/empty",
        2,
        "9b4fb24edd6d1d8830e272398263cdbf026b97392cc35387b991dc0248a628f9",
    ),
    (
        "zlib/0/one-byte",
        12,
        "fe97b7a0de74f93664f6a3a8a40c1785481602b36a49ce3a8ed78bfa01954296",
    ),
    (
        "deflate/0/one-byte",
        6,
        "0b96b18b4f7c1dcc466400c20c035f8542c8d19cecebe0e198c10be917b464a8",
    ),
    (
        "zlib/1/one-byte",
        9,
        "8697003ea5c8f20f906cf3e1c5a02071cd8030421091c335812e8c3555cd673c",
    ),
    (
        "deflate/1/one-byte",
        3,
        "59fced9a2ff83dabdd7055cf3abd6c12b802e6397b6fba5b8251cdf95d2bc991",
    ),
    (
        "zlib/6/one-byte",
        9,
        "8ecacf17e3a7f86d1b03a3012ce1270e80b2a8468ad3942580a4c85c6231414e",
    ),
    (
        "deflate/6/one-byte",
        3,
        "59fced9a2ff83dabdd7055cf3abd6c12b802e6397b6fba5b8251cdf95d2bc991",
    ),
    (
        "zlib/9/one-byte",
        9,
        "324b3f61040f09ab50dce3ff0cf1d3f19cbbe001e8360bdbf3ec4c04f2e4191b",
    ),
    (
        "deflate/9/one-byte",
        3,
        "59fced9a2ff83dabdd7055cf3abd6c12b802e6397b6fba5b8251cdf95d2bc991",
    ),
    (
        "zlib/0/english",
        2891,
        "a34702618783a9b0be7db99892eceda672a0a8cff822c791fdd0ea9426a4d84e",
    ),
    (
        "deflate/0/english",
        2885,
        "358455c41eb8cc81bc7a8568223e6fe8ee1cc8439663566a3a42bab79a6c92c4",
    ),
    (
        "zlib/1/english",
        87,
        "7dd57fa4c2e9bdeecd6c477a9fbdf183535fa1ab8037404768532736c7515200",
    ),
    (
        "deflate/1/english",
        81,
        "0a185f0f4308e3750ce45d91a021d5af307902bea18e45ddde73b48dbf8895c7",
    ),
    (
        "zlib/6/english",
        73,
        "8c4f9f134d75d0bdac5b3de8b0e72f023ce7fbf6ce406229fd13bfb6e4a53581",
    ),
    (
        "deflate/6/english",
        67,
        "ae5b8783b92710fdb6bd2d22d26bcc40b94069d84db81ade44f5e0e61170f94d",
    ),
    (
        "zlib/9/english",
        73,
        "d5d31c4e42e1bec30943d97ef60482b57fe870e0a80d6886f603ea1776bad8d6",
    ),
    (
        "deflate/9/english",
        67,
        "ae5b8783b92710fdb6bd2d22d26bcc40b94069d84db81ade44f5e0e61170f94d",
    ),
    (
        "zlib/0/japanese",
        2363,
        "0a0ec5bd626e80cff96aefedb3ae8d99813fe54977c5c9f3dcad6901c4b35c91",
    ),
    (
        "deflate/0/japanese",
        2357,
        "718f0c0650458a69d4703f185c9d458a4c9f4b58fc370cf6881449ce97d1c47d",
    ),
    (
        "zlib/1/japanese",
        86,
        "15b1cd5047fef14cb0c8e1894eb9dd5a991a5e14e161d963a5d9da5cb7de165e",
    ),
    (
        "deflate/1/japanese",
        80,
        "6dd240f924540d82bb776d98c075c7b0cda678561db821f3211b1de78242d579",
    ),
    (
        "zlib/6/japanese",
        78,
        "28a3ecb0877fd6a856cf8729de953fa1d54e18c862248df7e1c878c16e808e37",
    ),
    (
        "deflate/6/japanese",
        72,
        "b2b6be70eac689c4e6a8dd3166d9d9d0a0f4b087f5843cc010a63b8fb512ea20",
    ),
    (
        "zlib/9/japanese",
        78,
        "f4b2a79b46afc3d92cfbaf0703ff1081a130de008dedacfc6993a774d01df665",
    ),
    (
        "deflate/9/japanese",
        72,
        "b2b6be70eac689c4e6a8dd3166d9d9d0a0f4b087f5843cc010a63b8fb512ea20",
    ),
    (
        "zlib/0/ramp",
        4107,
        "2d8c19b8f2e96c73aa1510f3d6d3750e581e2c90b18af850ec9cb2e752a9256a",
    ),
    (
        "deflate/0/ramp",
        4101,
        "f4a6b313074fe2d8fe70d42c102dcd6f26338bd7076a122c33d372b5e6acc936",
    ),
    (
        "zlib/1/ramp",
        327,
        "df8b7354b3e603f6382f81b5fdff548577ac9e74522421a96c5725ba5ed1dfd0",
    ),
    (
        "deflate/1/ramp",
        321,
        "9c81bbcc62ca640df4393e7261b7fadef324aca5c95c3e83df38339c0db96ab3",
    ),
    (
        "zlib/6/ramp",
        315,
        "d28deb63dd99a7da5074070717d13e9dd487c6fefa144c9ccfead89eb35c08b2",
    ),
    (
        "deflate/6/ramp",
        309,
        "db4c45faf7dcd2cba7932215d64f02a0ac88ceec27985f8e9ed72ea0c29806db",
    ),
    (
        "zlib/9/ramp",
        315,
        "3919da5f0ca7d2f31b71db39f1ffc2b110e5f80c07bcfb6ee38b40bd7369fb2d",
    ),
    (
        "deflate/9/ramp",
        309,
        "db4c45faf7dcd2cba7932215d64f02a0ac88ceec27985f8e9ed72ea0c29806db",
    ),
    (
        "zlib/0/lcg-64k",
        65557,
        "2f105e923ec3b1edc1955f963caa353a2dab8bae31b38661f8d546976090c8cd",
    ),
    (
        "deflate/0/lcg-64k",
        65551,
        "2f60302383ba41d62c36d5019e34d71cc59217d2e7a49acbe1151185682cad32",
    ),
    (
        "zlib/1/lcg-64k",
        65601,
        "4681d52b8ba24750739c395c08ab1934b4c77ff04c3a1aa6ecc4f2044d5b87ed",
    ),
    (
        "deflate/1/lcg-64k",
        65595,
        "c9004c20b6a0785bc4197b704842b04fdaba15060a32df9a7195ac9d8366f8ab",
    ),
    (
        "zlib/6/lcg-64k",
        65557,
        "7d606691d1ff903e9a6276e467096794d6e2f76c6cceed104ec25bffe6f32689",
    ),
    (
        "deflate/6/lcg-64k",
        65551,
        "2f60302383ba41d62c36d5019e34d71cc59217d2e7a49acbe1151185682cad32",
    ),
    (
        "zlib/9/lcg-64k",
        65557,
        "f275779dac7c5e0e8229ecc0b63c8d083b46b1131040d8ed15d846d4da4f165f",
    ),
    (
        "deflate/9/lcg-64k",
        65551,
        "2f60302383ba41d62c36d5019e34d71cc59217d2e7a49acbe1151185682cad32",
    ),
];

#[cfg(feature = "lzma")]
const LZMA_OUTPUTS: &[(&str, usize, &str)] = &[
    (
        "lzma/empty",
        23,
        "b8d8686989e9c36d659dddb4f0c7e800c778cf973adf59fc3e1022bfeaac4874",
    ),
    (
        "lzma/one-byte",
        24,
        "6bc8cd39a5b1695cbcc813c36fd62dd301fa9108f14246ecad9118ce5bd29150",
    ),
    (
        "lzma/english",
        1606,
        "65b17b87462fb9b7cd69691d8531718ba5dc547fe59ec91483ab8776bf1ced06",
    ),
    (
        "lzma/japanese",
        1151,
        "118a13d3ca19b3380e0a176407ac750c31db01bd245f330f4ec13c9d4aa441f1",
    ),
    (
        "lzma/ramp",
        2824,
        "cbc345db16fe9b8ba5f4894296792c63941f3c45f9ad3676bd9f13a03958df26",
    ),
    (
        "lzma/lcg-64k",
        66463,
        "595d900199c248aea02d2fb819386884bf6241428e4426d17afb67a3fc9ef88f",
    ),
];

#[test]
fn probe_inputs_are_the_reference_bytes() {
    let rows: Vec<String> = reference()
        .into_iter()
        .filter(|l| l.starts_with("input "))
        .collect();
    let ours: Vec<String> = codec_probe_inputs()
        .iter()
        .map(|(name, b)| format!("input {name} {} {}", b.len(), hex(&Sha256::digest(b))))
        .collect();
    assert!(
        !rows.is_empty(),
        "reference.txt has no input rows: nothing compared"
    );
    assert_eq!(ours, rows);
}

#[test]
fn fold_is_the_reference_fold() {
    let vectors: Vec<(&[u8], Outputs)> = vec![
        (b"deflate", vec![]),
        (b"deflate", vec![("a".into(), vec![])]),
        (b"lzma", vec![("a".into(), b"bc".to_vec())]),
        (b"lzma", vec![("ab".into(), b"c".to_vec())]),
        (
            b"deflate",
            vec![("x".into(), vec![0, 1]), ("y".into(), vec![0xff])],
        ),
    ];
    let rows: Vec<String> = reference()
        .into_iter()
        .filter(|l| l.starts_with("fold "))
        .collect();
    let ours: Vec<String> = vectors
        .iter()
        .enumerate()
        .map(|(i, (codec, outs))| format!("fold {i} {}", hex(&fold_codec_outputs(codec, outs))))
        .collect();
    assert_eq!(
        rows.len(),
        vectors.len(),
        "reference.txt fold rows: nothing or too little compared"
    );
    assert_eq!(ours, rows);
}

fn dump(outputs: &[(String, Vec<u8>)]) {
    let Ok(dir) = std::env::var("ALICE_ZIP_DUMP_CODEC_OUTPUTS") else {
        return;
    };
    std::fs::create_dir_all(&dir).expect("create the dump directory");
    for (label, bytes) in outputs {
        let name = match label.split('/').collect::<Vec<_>>().as_slice() {
            [codec, level, probe] => format!("{codec}__{level}__{probe}"),
            [codec, probe] => format!("{codec}__-__{probe}"),
            _ => panic!("unexpected label {label}"),
        };
        std::fs::write(std::path::Path::new(&dir).join(name), bytes).expect("write a dump file");
    }
}

fn assert_recorded(outputs: &[(String, Vec<u8>)], recorded: &[(&str, usize, &str)]) {
    let ours: Vec<(String, usize, String)> = outputs
        .iter()
        .map(|(l, b)| (l.clone(), b.len(), hex(&Sha256::digest(b))))
        .collect();
    let rec: Vec<(String, usize, String)> = recorded
        .iter()
        .map(|(l, n, d)| ((*l).to_owned(), *n, (*d).to_owned()))
        .collect();
    for (a, b) in ours.iter().zip(&rec) {
        assert_eq!(a, b, "first differing output");
    }
    assert_eq!(ours.len(), rec.len(), "number of outputs");
}

#[test]
fn deflate_outputs_are_the_recorded_bytes_and_decode() {
    let outputs = deflate_probe_outputs().expect("encode");
    assert_eq!(
        outputs.len(),
        6 * 4 * 2,
        "6 probes x 4 levels x (zlib, raw deflate)"
    );
    dump(&outputs);
    let inputs = codec_probe_inputs();
    for (label, bytes) in &outputs {
        let probe = label.rsplit('/').next().unwrap();
        let input = &inputs.iter().find(|(n, _)| *n == probe).unwrap().1;
        let back = if label.starts_with("zlib/") {
            zlib_decompress(bytes).unwrap()
        } else {
            let mut v = Vec::new();
            flate2::read::DeflateDecoder::new(&bytes[..])
                .read_to_end(&mut v)
                .unwrap();
            v
        };
        assert_eq!(&back, input, "{label} does not decode to its input");
    }
    assert_recorded(&outputs, DEFLATE_OUTPUTS);
}

#[test]
fn deflate_id_is_the_recorded_value() {
    assert_eq!(hex(&deflate_bytes_id().unwrap()), hex(&DEFLATE_BYTES_ID));
}

/// Every output is load-bearing: leaving any one out, or changing one byte of it, moves the id
#[test]
fn every_deflate_output_moves_the_id() {
    let outputs = deflate_probe_outputs().unwrap();
    let id = fold_codec_outputs(b"deflate", &outputs);
    let mut compared = 0;
    for i in 0..outputs.len() {
        let mut fewer = outputs.clone();
        fewer.remove(i);
        assert_ne!(
            fold_codec_outputs(b"deflate", &fewer),
            id,
            "dropping {}",
            outputs[i].0
        );
        let mut changed = outputs.clone();
        match changed[i].1.first_mut() {
            Some(b) => *b ^= 1,
            None => changed[i].1.extend_from_slice(&[0]),
        }
        assert_ne!(
            fold_codec_outputs(b"deflate", &changed),
            id,
            "changing {}",
            outputs[i].0
        );
        compared += 1;
    }
    assert_eq!(compared, 48);
}

#[test]
fn check_codec_bytes_passes_in_this_build_and_compares_every_codec() {
    let compared = match check_codec_bytes() {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    };
    let expected: &[&str] = if cfg!(feature = "lzma") {
        &["deflate", "lzma"]
    } else {
        &["deflate"]
    };
    assert_eq!(compared, expected);
}

#[cfg(feature = "lzma")]
mod lzma {
    use super::*;
    use alice_zip::compression::{
        lzma_bytes_id, lzma_decompress, lzma_probe_outputs, LZMA_BYTES_ID,
    };

    #[test]
    fn lzma_outputs_are_the_recorded_bytes_and_decode() {
        let outputs = lzma_probe_outputs().expect("encode");
        assert_eq!(outputs.len(), 6);
        dump(&outputs);
        for ((label, bytes), (_, input)) in outputs.iter().zip(codec_probe_inputs()) {
            assert_eq!(lzma_decompress(bytes).unwrap(), input, "{label}");
        }
        assert_recorded(&outputs, LZMA_OUTPUTS);
    }

    #[test]
    fn lzma_id_is_the_recorded_value() {
        assert_eq!(hex(&lzma_bytes_id().unwrap()), hex(&LZMA_BYTES_ID));
    }
}
