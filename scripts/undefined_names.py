#!/usr/bin/env python3
"""Every tracked Python file is free of undefined names (F821) and of
redefinitions that hide an earlier definition (F811), checked with ruff.

A rename that leaves a call to the old name (a test runner, a helper) is
caught here; pytest does not run module-level `__main__` blocks, so it cannot.
A run that checked no file fails.

    python3 scripts/undefined_names.py
The ruff version is pinned (RUFF_VERSION); ruff is taken from PATH, from
`python -m ruff`, or run through `uvx`.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUFF_VERSION = "0.14.0"


def ruff_command() -> list:
    if shutil.which("ruff"):
        return ["ruff"]
    try:
        import ruff  # noqa: F401
        return [sys.executable, "-m", "ruff"]
    except ImportError:
        pass
    if shutil.which("uvx"):
        return ["uvx", f"ruff@{RUFF_VERSION}"]
    sys.exit("undefined-names: ruff not found (pip install ruff==" + RUFF_VERSION + ")")


def main() -> int:
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.split()
    if not files:
        print("undefined-names: no Python file checked")
        return 1
    version = subprocess.run(ruff_command() + ["--version"], capture_output=True, text=True)
    if RUFF_VERSION not in version.stdout:
        print(f"undefined-names: ruff {RUFF_VERSION} expected, got {version.stdout.strip()}")
        return 1
    r = subprocess.run(ruff_command() + ["check", "--select", "F821,F811", "--no-cache",
                                         "--output-format", "concise", *files], cwd=ROOT)
    if r.returncode != 0:
        return 1
    print(f"undefined-names: OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
