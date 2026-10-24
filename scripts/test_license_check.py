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
         "alice_zip/__init__.py", "LICENSE-Proprietary", "libalice/LICENSE-Proprietary",
         "README.md", "README_ja.md", "libalice/README.md", "deny.toml"]


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

    def test_a_new_manifest_with_another_licence_is_found(self):
        m = self.root / "wasm" / "Cargo.toml"
        m.parent.mkdir()
        m.write_text('[package]\nname = "x"\nlicense = "Apache-2.0"\npublish = false\n', encoding="utf-8")
        self.assertOneProblem("wasm/Cargo.toml: license")

    def test_tool_and_fuzz_manifests_are_not_package_licences(self):
        for d in ("tools/t", "fuzz"):
            m = self.root / d / "Cargo.toml"
            m.parent.mkdir(parents=True)
            m.write_text('[package]\nname = "t"\nlicense = "Apache-2.0"\npublish = false\n', encoding="utf-8")
        self.assertEqual(self.problems(), [])

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
        self.assertEqual(problems, [])
        self.assertGreaterEqual(compared, 30)

    def test_a_manifest_with_an_open_licence(self):
        self.edit("Cargo.toml", 'license = "LicenseRef-Proprietary"', 'license = "Apache-2.0"')
        self.assertOneProblem("Cargo.toml: license")

    def test_libalice_manifest(self):
        self.edit("libalice/Cargo.toml", 'license = "LicenseRef-Proprietary"', 'license = "MIT"')
        self.assertOneProblem("libalice/Cargo.toml: license")

    def test_a_manifest_without_publish_false(self):
        self.edit("libalice/Cargo.toml", "publish = false\n", "")
        self.assertOneProblem("libalice/Cargo.toml: publish = false")

    def test_pyproject_licence(self):
        self.edit("pyproject.toml", 'license = "LicenseRef-Proprietary"', 'license = "Apache-2.0"')
        self.assertOneProblem("pyproject.toml: license")

    def test_license_files_with_a_previous_file(self):
        self.edit("pyproject.toml", '["LICENSE-Proprietary"]', '["LICENSE-Proprietary", "NOTICE"]')
        self.assertOneProblem("license-files")

    def test_libalice_license_files_listed(self):
        self.edit("libalice/pyproject.toml", 'license = "LicenseRef-Proprietary"\n',
                  'license = "LicenseRef-Proprietary"\nlicense-files = ["LICENSE-Proprietary"]\n')
        self.assertOneProblem("license-files is set")

    def test_a_licence_classifier(self):
        self.edit("pyproject.toml", "classifiers = [\n",
                  'classifiers = [\n    "License :: OSI Approved :: Apache Software License",\n')
        self.assertOneProblem("classifier")

    def test_python_dunder_licence(self):
        self.edit("alice_zip/__init__.py", "__license__ = 'LicenseRef-Proprietary'", "__license__ = 'Apache-2.0'")
        self.assertOneProblem("__license__")

    def test_a_copy_that_differs(self):
        self.edit("libalice/LICENSE-Proprietary", "proprietary", "free")
        self.assertOneProblem("libalice/LICENSE-Proprietary")

    def test_a_missing_copy(self):
        (self.root / "libalice" / "LICENSE-Proprietary").unlink()
        self.assertOneProblem("libalice/LICENSE-Proprietary")

    def test_leftover_previous_licence_files(self):
        for rel in ["LICENSE-APACHE", "LICENSE-MIT", "NOTICE", "libalice/LICENSE-APACHE", "libalice/NOTICE"]:
            (self.root / rel).write_text("old", encoding="utf-8")
            self.assertOneProblem(f"{rel} is still present")
            (self.root / rel).unlink()

    def test_readme_badge(self):
        self.edit("README_ja.md", "license-LicenseRef--Proprietary-lightgrey", "license-Apache--2.0-green")
        self.assertOneProblem("badge")

    def test_readme_link_to_a_previous_file(self):
        self.edit("README.md", "[LICENSE-Proprietary](LICENSE-Proprietary).",
                  "[LICENSE-Proprietary](LICENSE-Proprietary) and [NOTICE](NOTICE).")
        self.assertOneProblem("links NOTICE")

    def test_readme_without_the_licence_link(self):
        self.edit("libalice/README.md", "[LICENSE-Proprietary](LICENSE-Proprietary)", "LICENSE-Proprietary")
        self.assertOneProblem("does not link LICENSE-Proprietary")

    def test_readme_licence_line(self):
        self.edit("README.md", "## License\n", "## License\n\nLicense: Apache-2.0\n")
        self.assertOneProblem("licence line")

    def test_a_stale_deny_comment(self):
        for words in ["MIT OR Apache-2.0", "Apache-2.0", "AGPL-3.0"]:
            self.edit("deny.toml", "# Licenses (allow-list", f"# alice-zip は {words}\n# Licenses (allow-list")
            self.assertOneProblem("deny.toml")
            self.edit("deny.toml", f"# alice-zip は {words}\n", "")


if __name__ == "__main__":
    unittest.main()
