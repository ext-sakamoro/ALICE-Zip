#!/usr/bin/env python3
"""scripts/lock_check.py fails on a lock file cargo would rewrite, and when it checked none.

    python3 scripts/test_lock_check.py
"""
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lock_check  # noqa: E402


def runner(locks, stale=()):
    calls = []

    def run(cmd):
        calls.append(cmd)
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, "\n".join(locks) + "\n", "")
        manifest = cmd[cmd.index("--manifest-path") + 1]
        lock = str(Path(manifest).parent / "Cargo.lock")
        if lock in stale:
            return subprocess.CompletedProcess(
                cmd, 101, "", f"error: cannot update the lock file {lock} because --locked was passed\n")
        return subprocess.CompletedProcess(cmd, 0, "{}", "")

    run.calls = calls
    return run


class LockCheck(unittest.TestCase):
    def test_matching_locks_pass(self):
        self.assertEqual(lock_check.check(runner(["Cargo.lock", "fuzz/Cargo.lock"])), ([], 2))

    def test_a_stale_lock_is_reported(self):
        problems, checked = lock_check.check(runner(["Cargo.lock", "fuzz/Cargo.lock"], stale={"fuzz/Cargo.lock"}))
        self.assertEqual(checked, 2)
        self.assertEqual(len(problems), 1)
        self.assertIn("fuzz/Cargo.lock", problems[0])

    def test_cargo_is_asked_with_locked_and_with_dependencies(self):
        r = runner(["sub/Cargo.lock"])
        lock_check.check(r)
        cargo = [c for c in r.calls if c[0] == "cargo"]
        self.assertEqual(len(cargo), 1)
        self.assertIn("--locked", cargo[0])
        self.assertNotIn("--no-deps", cargo[0])
        self.assertEqual(cargo[0][cargo[0].index("--manifest-path") + 1], "sub/Cargo.toml")

    def test_no_tracked_lock_checks_nothing(self):
        self.assertEqual(lock_check.check(runner([])), ([], 0))

    def test_main_fails_when_nothing_was_checked(self):
        orig = lock_check.run
        try:
            lock_check.run = runner([])
            self.assertEqual(lock_check.main(), 1)
            lock_check.run = runner(["Cargo.lock"], stale={"Cargo.lock"})
            self.assertEqual(lock_check.main(), 1)
            lock_check.run = runner(["Cargo.lock"])
            self.assertEqual(lock_check.main(), 0)
        finally:
            lock_check.run = orig


if __name__ == "__main__":
    unittest.main()
