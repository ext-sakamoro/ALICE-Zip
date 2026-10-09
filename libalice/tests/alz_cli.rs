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
