"""scripts/dist_license_check.py refuses an archive without the licence
files or the expression, and refuses to check nothing.

    python3 scripts/test_dist_license_check.py
"""
import io
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dist_license_check as d  # noqa: E402

META = "Metadata-Version: 2.4\nName: x\nLicense-Expression: Apache-2.0\nLicense-File: LICENSE-APACHE\nLicense-File: NOTICE\n"


def wheel(path, files=("LICENSE-APACHE", "NOTICE"), meta=META):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x-1.0.dist-info/METADATA", meta)
        for f in files:
            z.writestr(f"x-1.0.dist-info/licenses/{f}", "text")


def sdist(path, files=("LICENSE-APACHE", "NOTICE"), meta=META):
    with tarfile.open(path, "w:gz") as t:
        for name, text in [("PKG-INFO", meta)] + [(f, "text") for f in files]:
            data = text.encode()
            info = tarfile.TarInfo(f"x-1.0/{name}")
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


class DistLicenseCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_complete_archives_pass(self):
        wheel(self.dir / "a.whl")
        sdist(self.dir / "a.tar.gz")
        self.assertEqual(d.problems_in(str(self.dir / "a.whl")), [])
        self.assertEqual(d.problems_in(str(self.dir / "a.tar.gz")), [])
        self.assertEqual(d.main([str(self.dir / "*")]), 0)

    def test_a_missing_file(self):
        wheel(self.dir / "a.whl", files=("LICENSE-APACHE",))
        sdist(self.dir / "a.tar.gz", files=("NOTICE",))
        self.assertTrue(any("NOTICE is not in" in p for p in d.problems_in(str(self.dir / "a.whl"))))
        self.assertTrue(any("LICENSE-APACHE is not in" in p for p in d.problems_in(str(self.dir / "a.tar.gz"))))

    def test_metadata_without_the_expression_or_file_entry(self):
        wheel(self.dir / "a.whl", meta=META.replace("Apache-2.0", "MIT"))
        sdist(self.dir / "a.tar.gz", meta=META.replace("License-File: NOTICE\n", ""))
        self.assertTrue(any("License-Expression" in p for p in d.problems_in(str(self.dir / "a.whl"))))
        self.assertTrue(any("License-File: NOTICE" in p for p in d.problems_in(str(self.dir / "a.tar.gz"))))

    def test_checking_nothing_fails(self):
        self.assertEqual(d.main([]), 1)
        self.assertEqual(d.main([str(self.dir / "*.whl")]), 1)


if __name__ == "__main__":
    unittest.main()
