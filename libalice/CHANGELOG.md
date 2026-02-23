# Changelog

All notable changes to ALICE-Zip (libalice) will be documented in this file.

## [2.2.0] - 2026-02-23

### Added
- `generators::polynomial` — Polynomial evaluation (Horner's method) and least-squares fitting
- `generators::fourier` — FFT-based Fourier signal reconstruction and analysis via `rustfft`
- `generators::perlin` — Vectorized 2D Perlin noise generation with Rayon parallelism
- `analyzer` — Model competition: polynomial, Fourier, Perlin, constant, linear fitting
- `compression` — LZMA/zlib compression, 8-bit quantization, dequantization
- `residual` — Residual storage for lossless reconstruction
- `format` — `.alz` file format (32-byte header, multi-mode)
- `media` — Media-specific compression helpers
- `ffi` — C FFI exports
- `codec_bridge` — (feature `codec`) ALICE-Codec wavelet bridge
- Python bindings (feature `python`) — PyO3 + NumPy for all generators and analyzers
- CLI binary `alice` — compress, decompress, info, benchmark commands
- 81 unit tests
