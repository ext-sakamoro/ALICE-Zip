//! The `alice` command refuses an .alz file whose version or mode it never
//! wrote, instead of reading it as raw LZMA
//!
//! Runs the built binary (the production entry point) on a file it wrote and
//! on copies with one header byte changed.

use std::path::Path;
use std::process::Command;

fn compressed(dir: &Path) -> Vec<u8> {
    let input = dir.join("in.bin");
    std::fs::write(&input, b"some bytes to compress, some bytes to compress").unwrap();
    let out = dir.join("in.alz");
    let o = Command::new(env!("CARGO_BIN_EXE_alice"))
        .arg("compress")
        .arg(&input)
        .arg("-o")
        .arg(&out)
        .output()
        .expect("run alice compress");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    std::fs::read(&out).unwrap()
}

fn decompress(dir: &Path, name: &str, bytes: &[u8]) -> std::process::Output {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    let out = dir.join(format!("{name}.out"));
    Command::new(env!("CARGO_BIN_EXE_alice"))
        .arg("decompress")
        .arg(&path)
        .arg("-o")
        .arg(&out)
        .output()
        .expect("run alice decompress")
}

fn tempdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("alz-cli-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn a_file_the_command_wrote_reads_back() {
    let d = tempdir("ok");
    let alz = compressed(&d);
    let o = decompress(&d, "ok.alz", &alz);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn a_version_it_never_wrote_is_refused() {
    let d = tempdir("version");
    let mut alz = compressed(&d);
    alz[4] = 2;
    let o = decompress(&d, "v2.alz", &alz);
    assert!(!o.status.success(), "version 2 was read");
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("version"),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

#[test]
fn a_mode_it_never_wrote_is_refused() {
    let d = tempdir("mode");
    let mut alz = compressed(&d);
    alz[5] = 99;
    let o = decompress(&d, "m99.alz", &alz);
    assert!(!o.status.success(), "mode 99 was read as raw LZMA");
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("mode"),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

fn f32_input(dir: &Path) -> (std::path::PathBuf, Vec<f32>) {
    let vals: Vec<f32> = (0..4096)
        .map(|i| (f64::from(i) * 0.01).sin() as f32 * 3.0 + 0.25)
        .collect();
    let path = dir.join("in.f32");
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();
    (path, vals)
}

fn run(dir: &Path, args: &[&std::ffi::OsStr]) {
    let o = Command::new(env!("CARGO_BIN_EXE_alice"))
        .args(args)
        .current_dir(dir)
        .output()
        .expect("run alice");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn max_err(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

/// Half a quantisation step of `bits` over the range of `v`: the largest
/// error a correct quantiser of that width can make (plus f32 rounding).
fn half_step(v: &[f32], bits: u32) -> f64 {
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (f64::from(hi) - f64::from(lo)) / f64::from((1u32 << bits) - 1) / 2.0 * (1.0 + 1e-6) + 1e-6
}

#[test]
fn bits_16_keeps_16_bit_precision() {
    let d = tempdir("q16");
    let (input, vals) = f32_input(&d);
    let alz = d.join("q16.alz");
    let out = d.join("q16.out");
    run(
        &d,
        &[
            "compress".as_ref(),
            input.as_os_str(),
            "-o".as_ref(),
            alz.as_os_str(),
            "--bits".as_ref(),
            "16".as_ref(),
        ],
    );
    assert_eq!(
        std::fs::read(&alz).unwrap()[5],
        12,
        "16-bit data is written as mode 12"
    );
    run(
        &d,
        &[
            "decompress".as_ref(),
            alz.as_os_str(),
            "-o".as_ref(),
            out.as_os_str(),
        ],
    );
    let err = max_err(&vals, &read_f32(&out));
    assert!(
        err <= half_step(&vals, 16),
        "max error {err} exceeds a 16-bit half step {}",
        half_step(&vals, 16)
    );
}

#[test]
fn a_mode_11_file_from_earlier_releases_still_reads_as_8_bit() {
    // earlier releases wrote 8-bit data under mode 11 when asked for 16 bits
    let d = tempdir("legacy11");
    let (input, vals) = f32_input(&d);
    let alz = d.join("q8.alz");
    run(
        &d,
        &[
            "compress".as_ref(),
            input.as_os_str(),
            "-o".as_ref(),
            alz.as_os_str(),
            "--bits".as_ref(),
            "8".as_ref(),
        ],
    );
    let mut bytes = std::fs::read(&alz).unwrap();
    assert_eq!(bytes[5], 10);
    bytes[5] = 11;
    let legacy = d.join("legacy11.alz");
    std::fs::write(&legacy, &bytes).unwrap();
    let out = d.join("legacy11.out");
    run(
        &d,
        &[
            "decompress".as_ref(),
            legacy.as_os_str(),
            "-o".as_ref(),
            out.as_os_str(),
        ],
    );
    let err = max_err(&vals, &read_f32(&out));
    assert!(err <= half_step(&vals, 8), "max error {err}");
}
