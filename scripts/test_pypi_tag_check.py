"""scripts/pypi_tag_check.py picks the project from the tag and refuses a
version that differs from the package's, an unknown tag and a run that
compared nothing.

    python3 scripts/test_pypi_tag_check.py
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pypi_tag_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FILES = ["pyproject.toml", "libalice/Cargo.toml"]


class PypiTagCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for rel in FILES:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, self.root / rel)
        self.py = pypi_tag_check.table_version((ROOT / "pyproject.toml").read_text(encoding="utf-8"), "project")
        self.lib = pypi_tag_check.table_version(
            (ROOT / "libalice/Cargo.toml").read_text(encoding="utf-8"), "package"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def run_check(self, tag):
        return pypi_tag_check.check(tag, self.root)

    def test_the_repository_versions_are_read(self):
        self.assertRegex(self.py or "", r"^\d+\.\d+\.\d+")
        self.assertRegex(self.lib or "", r"^\d+\.\d+\.\d+")

    def test_matching_tags_pass_and_name_their_project(self):
        self.assertEqual(self.run_check(f"py-alice-zip-v{self.py}"), ("alice-zip", [], 1))
        self.assertEqual(self.run_check(f"py-libalice-v{self.lib}"), ("libalice", [], 1))

    def test_a_different_version_fails(self):
        project, problems, compared = self.run_check("py-alice-zip-v0.0.1")
        self.assertEqual((project, compared), ("alice-zip", 1))
        self.assertEqual(len(problems), 1)
        project, problems, _ = self.run_check("py-libalice-v0.0.1")
        self.assertEqual(project, "libalice")
        self.assertEqual(len(problems), 1)

    def test_the_libalice_tag_is_compared_with_cargo_not_pyproject(self):
        # the wheel's version is Cargo's; the root pyproject version must not pass
        if self.py != self.lib:
            _, problems, _ = self.run_check(f"py-libalice-v{self.py}")
            self.assertEqual(len(problems), 1)
        cargo = self.root / "libalice/Cargo.toml"
        cargo.write_text(
            cargo.read_text(encoding="utf-8").replace(f'version = "{self.lib}"', 'version = "9.9.9"', 1),
            encoding="utf-8",
        )
        self.assertEqual(self.run_check("py-libalice-v9.9.9"), ("libalice", [], 1))

    def test_crate_tags_and_unknown_tags_compare_nothing_and_fail(self):
        for tag in ("v0.9.0", "alice-zip-cli-v3.0.0", "py-v2.0.0", "py-alice-zip-2.0.0", ""):
            project, problems, compared = self.run_check(tag)
            self.assertIsNone(project, tag)
            self.assertEqual(compared, 0, tag)
            self.assertEqual(len(problems), 1, tag)

    def test_a_tag_without_a_version_fails(self):
        for tag in ("py-alice-zip-v", "py-libalice-vlatest"):
            _, problems, compared = self.run_check(tag)
            self.assertEqual(compared, 0, tag)
            self.assertEqual(len(problems), 1, tag)

    def test_a_missing_version_line_fails(self):
        p = self.root / "pyproject.toml"
        p.write_text(p.read_text(encoding="utf-8").replace(f'version = "{self.py}"', "", 1), encoding="utf-8")
        _, problems, compared = self.run_check(f"py-alice-zip-v{self.py}")
        self.assertEqual(compared, 0)
        self.assertEqual(len(problems), 1)

    def test_main_exit_codes(self):
        self.assertEqual(pypi_tag_check.main(["x", f"py-alice-zip-v{self.py}"]), 0)
        self.assertEqual(pypi_tag_check.main(["x", "py-alice-zip-v0.0.1"]), 1)
        self.assertEqual(pypi_tag_check.main(["x", "v0.9.0"]), 1)
        self.assertEqual(pypi_tag_check.main(["x"]), 2)


if __name__ == "__main__":
    unittest.main()
