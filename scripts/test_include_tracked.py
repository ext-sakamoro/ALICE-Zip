#!/usr/bin/env python3
"""Tests for include_tracked.py (each case builds a throwaway git repository)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import include_tracked  # noqa: E402


def repo(files, track):
    d = Path(tempfile.mkdtemp())
    subprocess.run(["git", "init", "-q", str(d)], check=True)
    for name, body in files.items():
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    subprocess.run(["git", "-C", str(d), "add", "--", *track], check=True)
    return d


class IncludeTracked(unittest.TestCase):
    def test_tracked_fixture_passes(self):
        d = repo({"tests/a.rs": b'const A: &[u8] = include_bytes!("data/x.bin");', "tests/data/x.bin": b"\0"},
                 ["tests/a.rs", "tests/data/x.bin"])
        self.assertEqual(include_tracked.check(d), (1, []))

    def test_untracked_fixture_fails(self):
        d = repo({"tests/a.rs": b'const A: &[u8] = include_bytes!("data/x.alice");', "tests/data/x.alice": b"\0"},
                 ["tests/a.rs"])
        compared, problems = include_tracked.check(d)
        self.assertEqual(compared, 1)
        self.assertEqual(len(problems), 1)
        self.assertIn("untracked", problems[0])

    def test_missing_fixture_fails(self):
        d = repo({"src/a.rs": b'const A: &str = include_str!("../gone.txt");'}, ["src/a.rs"])
        compared, problems = include_tracked.check(d)
        self.assertEqual((compared, "missing" in problems[0]), (1, True))

    def test_untracked_source_is_not_read(self):
        d = repo({"tests/a.rs": b"fn main() {}", "tests/b.rs": b'include_bytes!("no.bin");'}, ["tests/a.rs"])
        self.assertEqual(include_tracked.check(d), (0, []))

    def test_a_path_on_the_next_line_is_read(self):
        src = b'const A: &[u8] = include_bytes!(\n    "data/x.alice"\n);\nconst B: &str =\n    include_str!("data/y.txt");'
        d = repo({"tests/a.rs": src, "tests/data/x.alice": b"\0", "tests/data/y.txt": b"y"},
                 ["tests/a.rs", "tests/data/y.txt"])
        compared, problems = include_tracked.check(d)
        self.assertEqual(compared, 2)
        self.assertEqual(len(problems), 1)
        self.assertIn("tests/a.rs:1: tests/data/x.alice is untracked", problems[0])

    def test_zero_includes_fail_the_command(self):
        d = repo({"src/a.rs": b"fn main() {}"}, ["src/a.rs"])
        sys.argv = ["include_tracked.py", str(d)]
        self.assertEqual(include_tracked.main(), 1)


if __name__ == "__main__":
    unittest.main()
