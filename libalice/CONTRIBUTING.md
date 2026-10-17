# Contributing to ALICE-Zip (libalice)

## Build

```bash
cargo build
```

## Test

```bash
cargo test --lib --tests
```

## Lint

```bash
cargo clippy --lib --tests -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Optional Features

```bash
# Python bindings (requires Python environment)
cargo build --features python

# ALICE-Codec wavelet bridge
cargo build --features codec
```

## Design Constraints

- **Kolmogorov compression**: the shortest program that produces the output is the optimal representation.
- **Model competition**: polynomial, Fourier, Perlin, constant, linear models compete per segment.
- **Horner's method**: polynomial evaluation avoids explicit `pow()` calls.
- **Rayon parallelism**: Perlin noise generation is data-parallel across rows.
- **Reciprocal pre-computation**: divisions in hot loops replaced with multiplication by reciprocal.
- **FFT**: Fourier analysis uses the core crate's in-crate f64 FFT (radix-2, Bluestein for other lengths), the same bits on every target.
