#!/usr/bin/env bash
# scripts/preflight.sh — local reproduction of the CI gates before `git push`.
# Every command is the one CI runs. A step this file does not cover is a step
# that can only fail remotely — when a workflow step is added, add it here in
# the same commit. Run `--quick` before every push.
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the test / bench suites)
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate roots invalidates only this repo's fingerprints.
relint() { git ls-files | grep -E '(^|/)src/(lib|main)\.rs$' | xargs -r touch; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
has_toolchain() { rustup toolchain list | grep -q "^$1"; }

# Steps CI runs that this file cannot reproduce locally (they can only fail remotely):
#   - ci.yml:libalice-python:Build + install libalice (maturin) (no cargo / grep)
#   - ci.yml:libalice-python:Licence files in the wheels and sdists (needs maturin + build; scripts/dist_license_check.py)
#   - ci.yml:libalice-python:pytest (native accelerator enabled) (no cargo / grep)
#   - ci.yml:libalice-python:pytest (pure Python fallback) (no cargo / grep)
#   - security-audit.yml:audit:Install cargo-audit (needs network / runner-only)
#   - security-audit.yml:deny:Install cargo-deny (needs network / runner-only)
#   - security-audit.yml:coverage (job is continue-on-error: informational in CI)
#   - security-audit.yml:stub-guard:FFI panic isolation (every extern "C" fn is guarded) (no cargo / grep)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_lz77_roundtrip] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_lz77_roundtrip] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_lz77_roundtrip] (needs nightly + cargo-fuzz; runner-only)
#   - fuzz.yml:fuzz:Check the runs (crash or 0 runs = fail) [target=fuzz_lz77_roundtrip] (reads the run log)
#   - fuzz.yml:fuzz:Report crash [target=fuzz_lz77_roundtrip] (no cargo / grep)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_lz77_decode] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_lz77_decode] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_lz77_decode] (needs nightly + cargo-fuzz; runner-only)
#   - fuzz.yml:fuzz:Check the runs (crash or 0 runs = fail) [target=fuzz_lz77_decode] (reads the run log)
#   - fuzz.yml:fuzz:Report crash [target=fuzz_lz77_decode] (no cargo / grep)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_dictionary] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_dictionary] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_dictionary] (needs nightly + cargo-fuzz; runner-only)
#   - fuzz.yml:fuzz:Check the runs (crash or 0 runs = fail) [target=fuzz_dictionary] (reads the run log)
#   - fuzz.yml:fuzz:Report crash [target=fuzz_dictionary] (no cargo / grep)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_bpe] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_bpe] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_bpe] (needs nightly + cargo-fuzz; runner-only)
#   - fuzz.yml:fuzz:Check the runs (crash or 0 runs = fail) [target=fuzz_bpe] (reads the run log)
#   - fuzz.yml:fuzz:Report crash [target=fuzz_bpe] (no cargo / grep)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_zlib_roundtrip] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_zlib_roundtrip] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_zlib_roundtrip] (needs nightly + cargo-fuzz; runner-only)
#   - fuzz.yml:fuzz:Check the runs (crash or 0 runs = fail) [target=fuzz_zlib_roundtrip] (reads the run log)
#   - fuzz.yml:fuzz:Report crash [target=fuzz_zlib_roundtrip] (no cargo / grep)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_generators] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_generators] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_generators] (needs nightly + cargo-fuzz; runner-only)
#   - fuzz.yml:fuzz:Check the runs (crash or 0 runs = fail) [target=fuzz_generators] (reads the run log)
#   - fuzz.yml:fuzz:Report crash [target=fuzz_generators] (no cargo / grep)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_fourier_parity] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_fourier_parity] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_fourier_parity] (needs nightly + cargo-fuzz; runner-only)
#   - fuzz.yml:fuzz:Check the runs (crash or 0 runs = fail) [target=fuzz_fourier_parity] (reads the run log)
#   - fuzz.yml:fuzz:Report crash [target=fuzz_fourier_parity] (no cargo / grep)

need actionlint "brew install actionlint"
need cargo-audit "cargo install cargo-audit --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-hack "cargo install cargo-hack --locked"
need cargo-machete "cargo install cargo-machete --locked"
need cargo-semver-checks "cargo install cargo-semver-checks --version 0.50.0 --locked"
has_toolchain 1.87 || { echo "missing toolchain 1.87 (rustup toolchain install 1.87)" >&2; exit 1; }

step "ci.yml / clippy: clippy (default, all targets)"
relint
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo clippy --all-targets -- -D warnings )

step "ci.yml / clippy: clippy (docs.rs feature set, all targets)"
relint
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo clippy --all-targets --features "$ALL_FEATURES" -- -D warnings )

step "ci.yml / clippy: clippy (no default features, all targets)"
relint
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo clippy --all-targets --no-default-features -- -D warnings )

step "ci.yml / clippy: Clippy canary data (generator tests)"
python3 scripts/test_gen_clippy_canary_calls.py

step "ci.yml / clippy: Clippy canary data (generated from clippy.toml, no drift)"
python3 scripts/gen_clippy_canary_calls.py --check

step "ci.yml / clippy: Clippy canary (tests)"
python3 scripts/test_clippy_canary.py

step "ci.yml / clippy: libm ban list (tests)"
python3 scripts/test_libm_ban_check.py

step "ci.yml / no_std: Build rlib for a target without std"
rustup target list --installed | grep -q '^thumbv7em-none-eabihf$' || rustup target add thumbv7em-none-eabihf
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo rustc --lib --no-default-features --crate-type rlib --target thumbv7em-none-eabihf )

step "ci.yml / no_std: Clippy on the no_std build (-D warnings)"
relint
rustup target list --installed | grep -q '^thumbv7em-none-eabihf$' || rustup target add thumbv7em-none-eabihf
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; RUSTC_WORKSPACE_WRAPPER="$(rustup which clippy-driver)" cargo rustc --lib --no-default-features --crate-type rlib --target thumbv7em-none-eabihf -- -D warnings )

step "ci.yml / no_std: Host lib build without std (-D warnings)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; RUSTFLAGS="-D warnings" cargo build --lib --no-default-features )

step "ci.yml / no_std: Host lib build without std, alice-det-math std on (-D warnings)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; RUSTFLAGS="-D warnings" cargo build --lib --no-default-features --features alice-det-math/std )

step "ci.yml / msrv: cargo check on MSRV (lib, default features)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo +1.87 check --lib --locked )

step "ci.yml / msrv: cargo check on MSRV (lib, docs.rs feature set)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo +1.87 check --lib --locked --features "$ALL_FEATURES" )

step "ci.yml / msrv: cargo check on MSRV (lib, no_std)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo +1.87 check --lib --locked --no-default-features )

step "ci.yml / msrv: cargo check on MSRV (libalice, default features)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo +1.87 check --manifest-path libalice/Cargo.toml --locked )

step "ci.yml / feature-powerset: Powerset (all depths)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo hack check --lib --feature-powerset )

step "ci.yml / doc: cargo doc (default)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps )

step "ci.yml / doc: cargo doc (docs.rs feature set)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$ALL_FEATURES" )

step "ci.yml / fmt: Check formatting (core)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo fmt -- --check )

step "ci.yml / fmt: Check formatting (libalice)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo fmt --manifest-path libalice/Cargo.toml -- --check )

step "ci.yml / claim-check: claim-test marker が実在の試験を指しているか"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; python3 scripts/claim_check.py )

step "ci.yml / claim-check: README の英語版と日本語版の対応"
( python3 scripts/test_readme_parity_check.py && python3 scripts/readme_parity_check.py )

step "ci.yml / claim-check: Python の未定義の名前と上書き (ruff F821 / F811)"
( python3 scripts/undefined_names.py )

step "ci.yml / claim-check: include の参照先が git で追跡されているか"
( python3 scripts/test_include_tracked.py && python3 scripts/include_tracked.py )

step "ci.yml / claim-check: build した library を commit していないか"
( python3 scripts/test_tracked_binaries.py && python3 scripts/tracked_binaries.py )

step "ci.yml / test: Versions agree"
( python3 scripts/test_version_check.py && python3 scripts/version_check.py && python3 scripts/test_pypi_tag_check.py )

step "ci.yml / test: run_tests.sh (tests)"
python3 scripts/test_run_tests.py

step "ci.yml / test: Build configuration guard (test, then check)"
python3 scripts/test_build_config_guard.py
python3 scripts/build_config_guard.py
python3 scripts/test_build_flags_check.py

step "ci.yml / test: Tracked lock files match their manifests"
( python3 scripts/test_lock_check.py && python3 scripts/lock_check.py )

step "ci.yml / test: Licence agrees and its files are packaged"
(
  python3 scripts/test_license_check.py && python3 scripts/license_check.py
  python3 scripts/test_dist_license_check.py
  for m in Cargo.toml libalice/Cargo.toml; do
    list=$(cargo package --list --allow-dirty --manifest-path "$m")
    for f in LICENSE-APACHE NOTICE; do
      printf '%s\n' "$list" | grep -qx "$f" || { echo "$m: the package lacks $f"; exit 1; }
    done
    echo "$m: LICENSE-APACHE and NOTICE packaged"
  done
)

step "ci.yml / libalice: Clippy (-D warnings, all targets, codec)"
relint
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo clippy --manifest-path libalice/Cargo.toml --all-targets --features codec -- -D warnings )

step "ci.yml / libalice: Doc (-D warnings)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; RUSTDOCFLAGS="-Dwarnings" cargo doc --manifest-path libalice/Cargo.toml --lib --no-deps )

step "ci.yml / libalice: Release build (cdylib + CLI)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo build --manifest-path libalice/Cargo.toml --release )

step "ci.yml / libalice-python: cargo check / clippy (python feature)"
relint
(
  export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"
  cargo check --manifest-path libalice/Cargo.toml --features python
  cargo clippy --manifest-path libalice/Cargo.toml --features python --all-targets -- -D warnings
)

step "ci.yml / actionlint: actionlint"
actionlint .github/workflows/*.yml

step "ci.yml / actionlint: workflow-timeouts (every job has timeout-minutes)"
python3 scripts/test_workflow_timeouts.py
python3 scripts/workflow_timeouts.py

step "ci.yml / actionlint: workflow-concurrency (main is never cancelled)"
python3 scripts/test_workflow_concurrency.py
python3 scripts/workflow_concurrency.py

step "security-audit.yml / deny: cargo deny (core, all features)"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo deny --all-features check all )

step "security-audit.yml / deny: cargo deny (libalice)"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo deny --manifest-path libalice/Cargo.toml --config deny.toml --features codec check all )

step "security-audit.yml / deny: cargo deny (libalice-enterprise)"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo deny --manifest-path libalice-enterprise/Cargo.toml --config deny.toml check all )

step "security-audit.yml / unused-deps: cargo machete"
cargo machete

step "security-audit.yml / package-integrity: cargo package (.crate を作る)"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo package --no-verify --allow-dirty )

step "security-audit.yml / package-integrity: 展開して隔離 build"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -euo pipefail
  crate=$(find "${CARGO_TARGET_DIR:-target}/package" -maxdepth 1 -name '*.crate' | head -1)
  [ -n "$crate" ] || { echo "::error::.crate が生成されていない"; exit 1; }
  echo "packaged: $(basename "$crate") ($(du -h "$crate" | cut -f1))"
  tmp=$(mktemp -d)
  tar xzf "$crate" -C "$tmp"
  cd "$tmp"/*/
  # 親 workspace を継承すると member 扱いになり、境界を越えた file が
  # 再び見えてしまう (検査の意味が消える) ので単独 package として build
  printf '\n[workspace]\n' >> Cargo.toml
  cargo check --all-features
)

step "semver-checks.yml / semver-checks: semver gate (tests)"
python3 scripts/test_semver_gate.py

step "semver-checks.yml / semver-checks: semver gate (latest crates.io release, every consumer feature set)"
python3 scripts/semver_gate.py --config .github/semver-gate.toml --target-dir "${CARGO_TARGET_DIR:-target}/semver"

step "semver-checks.yml / semver-checks: semver gate (alice-zip-cli, latest crates.io release, every consumer feature set)"
python3 scripts/semver_gate.py --manifest libalice/Cargo.toml --config .github/semver-gate-cli.toml --target-dir "${CARGO_TARGET_DIR:-target}/semver-cli"

step "security-audit.yml / stub-guard: Detect todo! / unimplemented! / panic!(STUB)"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'todo!\(|unimplemented!\(|panic!\([^)]*STUB' \
    src/ libalice/src/ --include="*.rs" \
    --exclude-dir=bin \
    || true)
  if [ -n "$hits" ]; then
    echo "❌ Stub / unimplemented / STUB panic detected (production path):"
    echo "$hits"
    echo ""
    echo "Fix: use ? operator, Result<T, E>, or remove the stub."
    echo "  A stub must fail fast, not return a silent Ok."
    exit 1
  fi
  echo "✓ No todo! / unimplemented! / panic!(STUB) in src/ or libalice/src/"
)

step "security-audit.yml / stub-guard: Detect dbg!() residual"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rn 'dbg!(' src/ libalice/src/ --include="*.rs" || true)
  if [ -n "$hits" ]; then
    echo "❌ dbg!() macro left in production code:"
    echo "$hits"
    exit 1
  fi
  echo "✓ No dbg!()"
)

step "security-audit.yml / stub-guard: Entropy guard (test first)"
( python3 scripts/test_entropy_guard.py )

step "security-audit.yml / stub-guard: Entropy guard (no OS entropy or wall clock in production code)"
( python3 scripts/entropy_guard.py )

step "security-audit.yml / stub-guard: Detect TODO / FIXME / XXX / HACK (informational)"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'TODO|FIXME|XXX|HACK' src/ libalice/src/ --include="*.rs" || true)
  if [ -n "$hits" ]; then
    echo "::warning::TODO/FIXME/XXX/HACK found (informational, not blocking):"
    echo "$hits" | head -50
  else
    echo "✓ No TODO/FIXME/XXX/HACK"
  fi
)

step "fuzz.yml / fuzz: Test the run-count reader"
( python3 scripts/test_fuzz_runs.py )

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_lz77_roundtrip]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_lz77_roundtrip"
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Replay committed seeds (blocking) [target=fuzz_lz77_roundtrip]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    seeds="seeds/fuzz_lz77_roundtrip"
    if [ -d "$seeds" ] && [ -n "$(ls -A "$seeds")" ]; then
      for f in "$seeds"/*; do
        echo "replay: $f"
        cargo +nightly fuzz run "fuzz_lz77_roundtrip" "$f"
      done
    else
      echo "no committed seeds for fuzz_lz77_roundtrip"
    fi
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_lz77_decode]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_lz77_decode"
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Replay committed seeds (blocking) [target=fuzz_lz77_decode]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    seeds="seeds/fuzz_lz77_decode"
    if [ -d "$seeds" ] && [ -n "$(ls -A "$seeds")" ]; then
      for f in "$seeds"/*; do
        echo "replay: $f"
        cargo +nightly fuzz run "fuzz_lz77_decode" "$f"
      done
    else
      echo "no committed seeds for fuzz_lz77_decode"
    fi
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_dictionary]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_dictionary"
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Replay committed seeds (blocking) [target=fuzz_dictionary]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    seeds="seeds/fuzz_dictionary"
    if [ -d "$seeds" ] && [ -n "$(ls -A "$seeds")" ]; then
      for f in "$seeds"/*; do
        echo "replay: $f"
        cargo +nightly fuzz run "fuzz_dictionary" "$f"
      done
    else
      echo "no committed seeds for fuzz_dictionary"
    fi
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_bpe]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_bpe"
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Replay committed seeds (blocking) [target=fuzz_bpe]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    seeds="seeds/fuzz_bpe"
    if [ -d "$seeds" ] && [ -n "$(ls -A "$seeds")" ]; then
      for f in "$seeds"/*; do
        echo "replay: $f"
        cargo +nightly fuzz run "fuzz_bpe" "$f"
      done
    else
      echo "no committed seeds for fuzz_bpe"
    fi
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_zlib_roundtrip]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_zlib_roundtrip"
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Replay committed seeds (blocking) [target=fuzz_zlib_roundtrip]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    seeds="seeds/fuzz_zlib_roundtrip"
    if [ -d "$seeds" ] && [ -n "$(ls -A "$seeds")" ]; then
      for f in "$seeds"/*; do
        echo "replay: $f"
        cargo +nightly fuzz run "fuzz_zlib_roundtrip" "$f"
      done
    else
      echo "no committed seeds for fuzz_zlib_roundtrip"
    fi
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_generators]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_generators"
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Replay committed seeds (blocking) [target=fuzz_generators]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    seeds="seeds/fuzz_generators"
    if [ -d "$seeds" ] && [ -n "$(ls -A "$seeds")" ]; then
      for f in "$seeds"/*; do
        echo "replay: $f"
        cargo +nightly fuzz run "fuzz_generators" "$f"
      done
    else
      echo "no committed seeds for fuzz_generators"
    fi
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_fourier_parity]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_fourier_parity"
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Replay committed seeds (blocking) [target=fuzz_fourier_parity]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    seeds="seeds/fuzz_fourier_parity"
    if [ -d "$seeds" ] && [ -n "$(ls -A "$seeds")" ]; then
      for f in "$seeds"/*; do
        echo "replay: $f"
        cargo +nightly fuzz run "fuzz_fourier_parity" "$f"
      done
    else
      echo "no committed seeds for fuzz_fourier_parity"
    fi
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "public-text.yml / public-text: Public text (ALICE-Law checker at the pin; tree, then the commits)"
python3 scripts/test_public_text.py
scripts/public_text.sh

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test / bench suites skipped)"; exit 0
fi

step "ci.yml / clippy: Clippy canary (every planted banned call is flagged)"
python3 scripts/clippy_canary.py

step "ci.yml / clippy: libm ban list (complete for the resolved libm, and enforced)"
python3 scripts/libm_ban_check.py

step "ci.yml / example: Run the example the README numbers come from"
( set -o pipefail; cargo run --release --example compression_ratio --features lzma | tee /tmp/alice-zip-example.txt && python3 scripts/example_check.py /tmp/alice-zip-example.txt )

step "release-profile.yml / release-goldens: Bit-exact goldens (cargo test --release)"
( export CARGO_TERM_COLOR="always"; scripts/run_tests.sh analytic_law_id,float_edge_golden,determinism_golden,float_environment=2 -- cargo test --release --test analytic_law_id --test float_edge_golden --test determinism_golden --test float_environment )

step "release-profile.yml / release-goldens: Effective rustc flags (G14)"
python3 scripts/build_flags_check.py -- cargo check --release --lib
python3 scripts/build_flags_check.py -- cargo check --release --lib --all-features
python3 scripts/build_flags_check.py -- cargo check --release --lib --no-default-features

step "ci.yml / test: Test (default features)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; scripts/run_tests.sh determinism_golden,float_edge_golden,analytic_law_id,minmax_sign_golden,float_environment=2 -- cargo test )

step "ci.yml / test: Test (docs.rs feature set)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; scripts/run_tests.sh determinism_golden,float_edge_golden,analytic_law_id,minmax_sign_golden,float_environment=2 -- cargo test --features "$ALL_FEATURES" )

step "ci.yml / test: Test (no_std, lib unit tests)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo test --lib --no-default-features )

step "ci.yml / test: Determinism golden (no_std build, same digests as the std build)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; scripts/run_tests.sh determinism_golden -- cargo test --test determinism_golden --no-default-features )

step "ci.yml / test: Container (no_std build, same bytes and identifiers)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; scripts/run_tests.sh container_oracle -- cargo test --test container_oracle --no-default-features )

step "ci.yml / test: Container example (reads back, resolves references, refuses a changed byte)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo run --example container_roundtrip )

step "ci.yml / test: Feature-gated suites run on this OS"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; set -o pipefail; for t in residual_container_oracle container_oracle; do cargo test --features "$ALL_FEATURES" --test "$t" 2>&1 | tee "${CARGO_TARGET_DIR:-target}/gated-$t.log"; sed 's/\x1b\[[0-9;]*m//g; s/\x1b(B//g' "${CARGO_TARGET_DIR:-target}/gated-$t.log" | grep -E 'test result: ok\. [1-9][0-9]* passed'; done )

step "ci.yml / wasm: Container oracles under wasmtime"
rustup target list --installed | grep -q '^wasm32-wasip1$' || rustup target add wasm32-wasip1
command -v wasmtime >/dev/null || { echo "wasmtime is required for the wasm32-wasip1 step"; exit 1; }
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime; set -o pipefail; cargo test --target wasm32-wasip1 --test container_oracle 2>&1 | tee "${CARGO_TARGET_DIR:-target}/wasm-test.log"; sed 's/\x1b\[[0-9;]*m//g; s/\x1b(B//g' "${CARGO_TARGET_DIR:-target}/wasm-test.log" | grep -E 'test result: ok\. [1-9][0-9]* passed' )

step "ci.yml / libalice: Test (default)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo test --manifest-path libalice/Cargo.toml )

step "ci.yml / libalice: Test (codec bridge)"
( export CARGO_TERM_COLOR="always" ALL_FEATURES="std,fft,parallel,lzma"; cargo test --manifest-path libalice/Cargo.toml --features codec )

step "security-audit.yml / audit: cargo audit (core)"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo audit --deny yanked )

step "security-audit.yml / audit: cargo audit (libalice)"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo audit --deny yanked --file libalice/Cargo.lock )

step "security-audit.yml / audit: cargo audit (libalice-enterprise)"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo audit --deny yanked --file libalice-enterprise/Cargo.lock )

echo; echo "preflight OK"
