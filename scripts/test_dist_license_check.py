"""scripts/dist_license_check.py refuses an archive without the licence
file or the expression, or with a previous licence file, and refuses to
check nothing.

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

META = "Metadata-Version: 2.4\nName: x\nLicense-Expression: LicenseRef-Proprietary\nLicense-File: LICENSE-Proprietary\n"


def wheel(path, files=("LICENSE-Proprietary",), meta=META):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x-1.0.dist-info/METADATA", meta)
        for f in files:
            z.writestr(f"x-1.0.dist-info/licenses/{f}", "text")


def sdist(path, files=("LICENSE-Proprietary",), meta=META):
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
        wheel(self.dir / "a.whl", files=())
        sdist(self.dir / "a.tar.gz", files=())
        self.assertTrue(any("LICENSE-Proprietary is not in" in p for p in d.problems_in(str(self.dir / "a.whl"))))
        self.assertTrue(any("LICENSE-Proprietary is not in" in p for p in d.problems_in(str(self.dir / "a.tar.gz"))))

    def test_a_previous_licence_file(self):
        old = META + "License-File: NOTICE\n"
        wheel(self.dir / "a.whl", files=("LICENSE-Proprietary", "LICENSE-APACHE"))
        sdist(self.dir / "a.tar.gz", meta=old, files=("LICENSE-Proprietary", "NOTICE"))
        self.assertTrue(any("LICENSE-APACHE is still in" in p for p in d.problems_in(str(self.dir / "a.whl"))))
        p = d.problems_in(str(self.dir / "a.tar.gz"))
        self.assertTrue(any("NOTICE is still in" in x for x in p), p)
        self.assertTrue(any("still names 'License-File: NOTICE'" in x for x in p), p)

    def test_metadata_without_the_expression_or_file_entry(self):
        wheel(self.dir / "a.whl", meta=META.replace("LicenseRef-Proprietary", "Apache-2.0"))
        sdist(self.dir / "a.tar.gz", meta=META.replace("License-File: LICENSE-Proprietary\n", ""))
        self.assertTrue(any("License-Expression" in p for p in d.problems_in(str(self.dir / "a.whl"))))
        self.assertTrue(any("License-File: LICENSE-Proprietary" in p for p in d.problems_in(str(self.dir / "a.tar.gz"))))

    def test_checking_nothing_fails(self):
        self.assertEqual(d.main([]), 1)
        self.assertEqual(d.main([str(self.dir / "*.whl")]), 1)


if __name__ == "__main__":
    unittest.main()
