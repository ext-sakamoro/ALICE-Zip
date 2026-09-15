#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push` (ci.yml + the blocking
# jobs of security-audit.yml). Every command is the one CI runs; a step this
# script does not cover is a step that can only fail remotely
# (feedback_ci_local_verify_preflight_2026_09_15).
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the test suites and fuzz build)
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1
ALL_FEATURES='std,fft,parallel'
MSRV=1.87

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

step "actionlint (workflow YAML)"
if have actionlint; then actionlint .github/workflows/*.yml; else echo "skip: actionlint not installed" >&2; fi

step "cargo fmt --check (core + libalice)"
cargo fmt -- --check
cargo fmt --manifest-path libalice/Cargo.toml -- --check

step "clippy -D warnings (default / docs.rs feature set, all targets)"
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features "$ALL_FEATURES" -- -D warnings

step "no_std rlib on thumbv7em-none-eabihf (+ clippy)"
cargo rustc --lib --no-default-features --crate-type rlib --target thumbv7em-none-eabihf
RUSTC_WORKSPACE_WRAPPER="$(rustup which clippy-driver)" \
  cargo rustc --lib --no-default-features --crate-type rlib --target thumbv7em-none-eabihf -- -D warnings

step "MSRV $MSRV (lib: default / docs.rs set / no_std)"
if rustup run "$MSRV" cargo --version >/dev/null 2>&1; then
  cargo "+$MSRV" check --lib --locked
  cargo "+$MSRV" check --lib --locked --features "$ALL_FEATURES"
  cargo "+$MSRV" check --lib --locked --no-default-features
  cargo "+$MSRV" check --manifest-path libalice/Cargo.toml --locked
else
  echo "skip: toolchain $MSRV not installed (rustup toolchain install $MSRV --profile minimal)" >&2
fi

step "feature powerset (cargo-hack)"
if have cargo-hack; then cargo hack check --lib --feature-powerset; else echo "skip: cargo-hack not installed" >&2; fi

step "rustdoc -D warnings (default / docs.rs set / libalice)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --features "$ALL_FEATURES"
RUSTDOCFLAGS="-Dwarnings" cargo doc --manifest-path libalice/Cargo.toml --lib --no-deps

step "libalice clippy -D warnings (codec feature / python feature)"
cargo clippy --manifest-path libalice/Cargo.toml --all-targets --features codec -- -D warnings
cargo check --manifest-path libalice/Cargo.toml --features python
cargo clippy --manifest-path libalice/Cargo.toml --all-targets --features python -- -D warnings

step "security: audit (3 lockfiles) / deny / machete / stub-guard / FFI guard"
if have cargo-audit; then
  cargo audit --deny yanked
  cargo audit --deny yanked --file libalice/Cargo.lock
  cargo audit --deny yanked --file libalice-enterprise/Cargo.lock
else echo "skip: cargo-audit not installed" >&2; fi
if have cargo-deny; then
  cargo deny --all-features check all
  cargo deny --manifest-path libalice/Cargo.toml --config deny.toml --features codec check all
  cargo deny --manifest-path libalice-enterprise/Cargo.toml --config deny.toml check all
else echo "skip: cargo-deny not installed" >&2; fi
if have cargo-machete; then cargo machete; else echo "skip: cargo-machete not installed" >&2; fi
hits=$(grep -rnE 'todo!\(|unimplemented!\(|panic!\([^)]*STUB|dbg!\(' src/ libalice/src/ --include="*.rs" --exclude-dir=bin || true)
[[ -z "$hits" ]] || { echo "stub / dbg residual:"; echo "$hits"; exit 1; }
python3 - <<'PY'
import re, sys, pathlib
bad = []
for f in pathlib.Path("libalice/src").rglob("*.rs"):
    s = f.read_text()
    for m in re.finditer(r'extern "C" fn (\w+)\([^{]*\{\n(.*?)\n', s, re.S):
        first = m.group(2).strip()
        if not (first.startswith("guarded(") or first.startswith("guarded_or(") or first.startswith("static ")):
            bad.append(f"{f}: {m.group(1)}: {first}")
if bad:
    print("extern \"C\" fn without catch_unwind guard:\n" + "\n".join(bad)); sys.exit(1)
print("FFI guard: ok")
PY

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick: OK (test suites / fuzz build / semver skipped)"; exit 0
fi

step "tests (default / docs.rs set / no_std lib / libalice)"
cargo test
cargo test --features "$ALL_FEATURES"
cargo test --lib --no-default-features
cargo test --manifest-path libalice/Cargo.toml
cargo test --manifest-path libalice/Cargo.toml --features codec

step "Python: maturin develop + pytest (native / pure), needs \$ALICE_ZIP_VENV or skips"
if [[ -n "${ALICE_ZIP_VENV:-}" && -x "$ALICE_ZIP_VENV/bin/python" ]]; then
  # shellcheck disable=SC1091
  . "$ALICE_ZIP_VENV/bin/activate"
  (cd libalice && maturin develop --release -q)
  pip install -q -e . && python -m pytest tests/ -q -p no:cacheprovider
  pip uninstall -q -y libalice && python -m pytest tests/ -q -p no:cacheprovider
  deactivate
else
  echo "skip: set ALICE_ZIP_VENV=<venv with maturin numpy pytest> to run the Python suite" >&2
fi

step "libalice release build (cdylib + CLI)"
cargo build --manifest-path libalice/Cargo.toml --release

step "fuzz targets build (nightly)"
if have cargo-fuzz && rustup run nightly cargo --version >/dev/null 2>&1; then
  (cd fuzz && cargo +nightly fuzz build)
else echo "skip: cargo-fuzz / nightly not installed" >&2; fi

step "semver-checks vs crates.io (docs.rs feature set)"
if have cargo-semver-checks; then
  cargo semver-checks check-release --package alice-zip --only-explicit-features --features "$ALL_FEATURES"
else echo "skip: cargo-semver-checks not installed" >&2; fi

echo; echo "preflight: OK"
