"""scripts/license_check.py refuses each kind of disagreement.

    python3 scripts/test_license_check.py
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import license_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Cargo.toml", "libalice/Cargo.toml", "pyproject.toml", "libalice/pyproject.toml",
         "alice_zip/__init__.py", "LICENSE-APACHE", "NOTICE", "libalice/LICENSE-APACHE",
         "libalice/NOTICE", "README.md", "README_ja.md", "libalice/README.md"]


class LicenseCheck(unittest.TestCase):
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
        p.write_text(text.replace(old, new, 1), encoding="utf-8")

    def problems(self):
        *problems, compared = license_check.check(self.root)
        self.assertGreater(compared, 0)
        return problems

    def assertOneProblem(self, fragment):
        problems = self.problems()
        self.assertEqual(len(problems), 1, problems)
        self.assertIn(fragment, problems[0])

    def test_the_repository_agrees(self):
        *problems, compared = license_check.check(self.root)
        self.assertEqual((problems, compared), ([], 21))

    def test_a_manifest_with_another_licence(self):
        self.edit("Cargo.toml", 'license = "Apache-2.0"', 'license = "MIT OR Apache-2.0"')
        self.assertOneProblem("Cargo.toml: license")

    def test_libalice_manifest(self):
        self.edit("libalice/Cargo.toml", 'license = "Apache-2.0"', 'license = "MIT"')
        self.assertOneProblem("libalice/Cargo.toml: license")

    def test_pyproject_licence_and_files(self):
        self.edit("pyproject.toml", 'license = "Apache-2.0"', 'license = "MIT"')
        self.assertOneProblem("pyproject.toml: license")

    def test_license_files_without_notice(self):
        self.edit("pyproject.toml", '"LICENSE-APACHE", "NOTICE"', '"LICENSE-APACHE"')
        self.assertOneProblem("license-files")

    def test_libalice_license_files_listed(self):
        self.edit("libalice/pyproject.toml", 'license = "Apache-2.0"\n',
                  'license = "Apache-2.0"\nlicense-files = ["LICENSE-APACHE", "NOTICE"]\n')
        self.assertOneProblem("license-files is set")

    def test_a_licence_classifier(self):
        self.edit("pyproject.toml", "classifiers = [\n",
                  'classifiers = [\n    "License :: OSI Approved :: MIT License",\n')
        self.assertOneProblem("classifier")

    def test_python_dunder_licence(self):
        self.edit("alice_zip/__init__.py", "__license__ = 'Apache-2.0'", "__license__ = 'MIT'")
        self.assertOneProblem("__license__")

    def test_a_copy_that_differs(self):
        self.edit("libalice/NOTICE", "ALICE-Zip", "ALICE-Zap")
        self.assertOneProblem("libalice/NOTICE")

    def test_a_missing_copy(self):
        (self.root / "libalice" / "LICENSE-APACHE").unlink()
        self.assertOneProblem("libalice/LICENSE-APACHE")

    def test_a_leftover_mit_file(self):
        (self.root / "LICENSE-MIT").write_text("MIT", encoding="utf-8")
        self.assertOneProblem("LICENSE-MIT is still present")

    def test_notice_without_attribution(self):
        self.edit("NOTICE", "This product includes", "This includes")
        self.edit("libalice/NOTICE", "This product includes", "This includes")
        self.assertOneProblem("attribution")

    def test_readme_badge(self):
        self.edit("README_ja.md", "license-Apache--2.0-green", "license-MIT-green")
        self.assertOneProblem("badge")

    def test_readme_link(self):
        self.edit("README.md", "[LICENSE-APACHE](LICENSE-APACHE)", "[LICENSE-MIT](LICENSE-MIT)")
        self.assertOneProblem("links LICENSE-MIT")


if __name__ == "__main__":
    unittest.main()
