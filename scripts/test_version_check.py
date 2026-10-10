"""scripts/version_check.py refuses each kind of disagreement.

    python3 scripts/test_version_check.py
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import version_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FILES = ["pyproject.toml", "libalice/pyproject.toml", "alice_zip/__init__.py"]


class VersionCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for rel in FILES:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, self.root / rel)

    def tearDown(self):
        self._tmp.cleanup()

    def edit(self, rel, old, new):
        p = self.root / rel
        text = p.read_text(encoding="utf-8")
        self.assertIn(old, text)
        p.write_text(text.replace(old, new), encoding="utf-8")

    def run_check(self):
        *problems, compared = version_check.check(self.root)
        return problems, compared

    def test_the_repository_agrees(self):
        self.assertEqual(self.run_check(), ([], 2))

    def test_a_static_libalice_version_is_refused(self):
        self.edit("libalice/pyproject.toml", 'dynamic = ["version"]', 'version = "2.4.0"')
        problems, _ = self.run_check()
        self.assertTrue(any("static version" in x for x in problems), problems)
        self.assertTrue(any("not in dynamic" in x for x in problems), problems)

    def test_a_python_version_that_differs_is_refused(self):
        self.edit("alice_zip/__init__.py", "__version__ = '", "__version__ = '9")
        problems, _ = self.run_check()
        self.assertTrue(any("!= pyproject.toml version" in x for x in problems), problems)


if __name__ == "__main__":
    unittest.main()
