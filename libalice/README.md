# libalice — native accelerator for ALICE-Zip

Rust crate `alice-zip-cli` (this directory) builds three things from one source:

| artifact | what | how |
|----------|------|-----|
| `alice` binary | CLI: `compress` / `decompress` / `info` / `benchmark` for `.alice` containers | `cargo build --release` |
| `libalice_core.{so,dylib,dll}` | C FFI (`include/alice.h`) used by the C++ / C# (Unity) / UE5 bindings in `../bindings/` | `cargo build --release` |
| `libalice` Python module | PyO3 extension imported by the `alice-zip` Python package (`alice_zip/native_accelerator.py`) as an optional speed-up | `maturin develop --release` / `maturin build --release` |

The generator laws (polynomial / Fourier / Perlin) are not implemented here: they
are re-exported from the core crate at the repository root
([`alice-zip` on crates.io](https://crates.io/crates/alice-zip)), see
`src/generators/mod.rs` for the conventions this crate pins (`.alice` container:
descending polynomial coefficients on `x ∈ [0, 1]`, rustfft analysis).

## Python

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin numpy
cd libalice && maturin develop --release        # installs `libalice` into the venv
python -c "import libalice; print(libalice.sine_wave(4, 1.0))"
```

Without the module installed the Python package falls back to pure Python
(same results, slower).

## FFI

Every `extern "C"` entry point runs inside `catch_unwind`; a Rust panic is
returned as `ALICE_ERROR_INTERNAL_PANIC` (7) with the message available from
`alice_get_last_error()` instead of aborting the host process. Buffers returned
by the library are freed with `alice_free_buffer` / `alice_free_float_buffer`
(a second free is a no-op).

## Versioning / license

`2.3.0` — see [CHANGELOG.md](CHANGELOG.md). MIT (`../LICENSE-MIT`).
