#!/usr/bin/env python3
"""Tests for scripts/libm_ban_check.py (no cargo: synthetic libm sources and
clippy.toml texts, plus the probe's verdict on made-up diagnostics)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libm_ban_check as m  # noqa: E402

MATH_MOD = """\
mod sin;
pub use self::ceil::{ceil, ceilf};
pub use self::floor::{floor, floorf};
pub use self::fma::{fma, fmaf};
pub use self::round::{round, roundf};
pub use self::sqrt::{sqrt, sqrtf};
pub use self::trunc::{trunc, truncf};
pub use self::sin::sin;
pub use self::sincos::sincos;
cfg_if! {
    if #[cfg(f16_enabled)] {
        pub use self::ceil::ceilf16;
        pub use self::nested::{a, b};
    }
}
"""
LIB_RS = "mod math;\npub use libm_helper::*;\npub use self::math::*;\n"


def toml(methods: list[str], types: tuple[str, ...] = ("libm::Libm",)) -> dict[str, list[str]]:
    return {"disallowed-methods": methods, "disallowed-types": list(types)}


class Parse(unittest.TestCase):
    def test_exports_skip_cfg_if_blocks(self):
        names = m.exported_functions(MATH_MOD)
        self.assertIn("sin", names)
        self.assertIn("sincos", names)
        self.assertNotIn("ceilf16", names)
        self.assertNotIn("a", names)
        self.assertEqual(len(names), 14)

    def test_unbalanced_cfg_if_is_an_error(self):
        with self.assertRaises(ValueError):
            m.exported_functions("cfg_if! {\n pub use self::x::y;\n")

    def test_unknown_lib_surface(self):
        self.assertEqual(m.unknown_lib_surface(LIB_RS), [])
        self.assertEqual(m.unknown_lib_surface(LIB_RS + "pub fn extra() {}\n"), ["pub fn extra() {}"])

    def test_clippy_toml_paths(self):
        text = ('disallowed-methods = [\n    # a comment with path = "x::y"\n'
                '    { path = "libm::sin", reason = "r" },\n]\n'
                'disallowed-types = [\n    { path = "libm::Libm", reason = "r" },\n]\n')
        self.assertEqual(m.clippy_toml_paths(text),
                         {"disallowed-methods": ["libm::sin"], "disallowed-types": ["libm::Libm"]})


class Check(unittest.TestCase):
    exports = sorted(m.ALLOWED) + ["sin", "sincos"]

    def test_complete_list_passes(self):
        self.assertEqual(m.check(self.exports, [], toml(["libm::sin", "libm::sincos", "f64::sin"])), [])

    def test_no_exports_compares_nothing(self):
        p = m.check([], [], toml(["libm::sin"]))
        self.assertEqual(len(p), 1)
        self.assertIn("compared nothing", p[0])

    def test_no_libm_entry_compares_nothing(self):
        p = m.check(self.exports, [], toml(["f64::sin"]))
        self.assertTrue(any("compared nothing" in x for x in p))

    def test_unbanned_export_fails(self):
        p = m.check(self.exports, [], toml(["libm::sin"]))
        self.assertEqual(p, ["libm::sincos is public in the resolved libm but neither allowed nor banned in clippy.toml"])

    def test_banned_name_that_does_not_exist_fails(self):
        p = m.check(self.exports, [], toml(["libm::sin", "libm::sincos", "libm::no_such"]))
        self.assertEqual(p, ["clippy.toml bans libm::no_such, which the resolved libm does not export"])

    def test_banning_an_allowed_name_fails(self):
        p = m.check(self.exports, [], toml(["libm::sin", "libm::sincos", "libm::sqrt"]))
        self.assertEqual(p, ["clippy.toml bans libm::sqrt, which is in the allowed set"])

    def test_allowed_name_missing_from_libm_fails(self):
        exports = [x for x in self.exports if x != "fmaf"]
        p = m.check(exports, [], toml(["libm::sin", "libm::sincos"]))
        self.assertEqual(p, ["allowed libm::fmaf is not exported by the resolved libm"])

    def test_duplicate_entry_fails(self):
        p = m.check(self.exports, [], toml(["libm::sin", "libm::sincos", "libm::sin"]))
        self.assertTrue(any("more than once" in x for x in p))

    def test_missing_helper_type_fails(self):
        p = m.check(self.exports, [], toml(["libm::sin", "libm::sincos"], types=()))
        self.assertEqual(p, ["libm::Libm is not listed under disallowed-types in clippy.toml"])

    def test_unknown_surface_fails(self):
        p = m.check(self.exports, ["pub mod extra;"], toml(["libm::sin", "libm::sincos"]))
        self.assertTrue(any("does not know" in x for x in p))


class Probe(unittest.TestCase):
    good = [(1, "use of a disallowed method `libm::sin`"),
            (2, "use of a disallowed method `libm::cosf`"),
            (3, "use of a disallowed method `libm::log2`"),
            (4, "use of a disallowed type `libm::Libm`")]

    def test_all_flagged_passes(self):
        self.assertEqual(m.probe_problems(101, self.good), [])

    def test_exit_zero_fails(self):
        self.assertTrue(any("exited 0" in x for x in m.probe_problems(0, self.good)))

    def test_missing_diagnostic_fails(self):
        p = m.probe_problems(101, self.good[1:])
        self.assertEqual(p, ["probe: line 1 got 0 disallowed diagnostic(s), 0 naming `libm::sin` (expected exactly 1)"])

    def test_wrong_path_fails(self):
        diags = [self.good[0], (2, "use of a disallowed method `libm::sinf`"), *self.good[2:]]
        self.assertTrue(any("line 2" in x for x in m.probe_problems(101, diags)))

    def test_two_diagnostics_on_one_line_fail(self):
        self.assertTrue(any("line 1 got 2" in x for x in m.probe_problems(101, self.good + [self.good[0]])))

    def test_flagged_control_fails(self):
        p = m.probe_problems(101, self.good + [(5, "use of a disallowed method `libm::sqrt`")])
        self.assertTrue(any("control" in x for x in p))

    def test_stray_diagnostic_fails(self):
        self.assertTrue(any("outside" in x for x in m.probe_problems(101, self.good + [(9, "x")])))


class Main(unittest.TestCase):
    def run_main(self, math_mod: str, clippy: str) -> int:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "libm")
            (src / "src" / "math").mkdir(parents=True)
            (src / "src" / "math" / "mod.rs").write_text(math_mod, encoding="utf-8")
            (src / "src" / "lib.rs").write_text(LIB_RS, encoding="utf-8")
            root = Path(d, "root")
            root.mkdir()
            (root / "clippy.toml").write_text(clippy, encoding="utf-8")
            return m.main(["--root", str(root), "--libm-src", str(src), "--no-probe"])

    def test_static_pass(self):
        clippy = ('disallowed-methods = [\n    { path = "libm::sin", reason = "r" },\n'
                  '    { path = "libm::sincos", reason = "r" },\n]\n'
                  'disallowed-types = [\n    { path = "libm::Libm", reason = "r" },\n]\n')
        self.assertEqual(self.run_main(MATH_MOD, clippy), 0)

    def test_empty_libm_source_fails(self):
        clippy = 'disallowed-methods = [\n    { path = "libm::sin", reason = "r" },\n]\n'
        self.assertEqual(self.run_main("", clippy), 1)


if __name__ == "__main__":
    unittest.main()
