"""The arithmetic identifiers of the Python package (alice_zip.semantics) and the
files that record them: the procedural ALICE_ZIP payload, the ALICE_IMG / AUD /
VID parameter JSON and the residual files (versions 5 to 7).

Expected values come from the generators' own output for the pinned inputs and
from the files the Rust writer wrote (tests/data/residual/rust_semantics.bin),
never from the reader under test.
"""
import hashlib
import json
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alice_zip import semantics  # noqa: E402
from alice_zip import native_accelerator as na  # noqa: E402
from alice_zip.core import ALICEZip  # noqa: E402
from alice_zip.media_generators import AudioGenerator, AudioParams, AudioPattern  # noqa: E402
from alice_zip.residual_compression import (  # noqa: E402
    ResidualCompressionMethod as M, ResidualCompressor, ResidualData)

DATA = ROOT / "tests" / "data" / "residual"
# the recorded value of the fold (changes only when a pin changes)
PYTHON_SEMANTICS_HEX = "ab369056f6db6b3f556a54fdfc186cf2ca9e8944dedfcdb4444fb1790f301aa0"


def f32_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).astype("<f4").tobytes()).hexdigest()


def header(b):
    n = struct.unpack("<I", b[:4])[0]
    return json.loads(b[4:4 + n])


# ------------------------------------------------------------------ the identifier

def test_every_output_pin_is_the_generators_output():
    pins = semantics.PYTHON_SEMANTICS_PINS
    got = {
        "golden python sine_wave": f32_digest(na._python_sine_wave(4096, 7.0, 1.5, 0.25, -0.5)),
        "golden python fourier_generate": f32_digest(na._python_fourier_generate(
            4096, [(3, 1.0, 0.1), (17, 0.5, 1.2), (200, 0.25, -0.7)], 0.125)),
        "golden python polynomial_generate": f32_digest(
            na._python_polynomial_generate(4096, [0.5, -1.25, 2.0, 0.75])),
        "golden python perlin_advanced": f32_digest(
            na._python_perlin_advanced(64, 48, 1234, 8.0, 4, 0.5, 2.0)),
    }
    for name, digest in got.items():
        assert pins[name] == digest, name
    # the white-noise pin is the golden the audio test records
    src = (ROOT / "tests" / "test_media_generators.py").read_text(encoding="utf-8")
    assert f'WHITE_NOISE_GOLDEN = "{pins["golden python white_noise"]}"' in src
    assert len(pins) == len(got) + 2


def test_the_identifier_is_the_fold_of_the_pins_and_moves_with_each():
    assert semantics.PYTHON_SEMANTICS_ID.hex() == PYTHON_SEMANTICS_HEX
    # the fold, written out here: names sorted, u32 big-endian length, name, digest
    h = hashlib.sha256()
    for name in sorted(semantics.PYTHON_SEMANTICS_PINS):
        raw = name.encode()
        h.update(struct.pack(">I", len(raw)) + raw + bytes.fromhex(semantics.PYTHON_SEMANTICS_PINS[name]))
    assert h.digest() == semantics.PYTHON_SEMANTICS_ID
    for name in semantics.PYTHON_SEMANTICS_PINS:
        changed = dict(semantics.PYTHON_SEMANTICS_PINS)
        changed[name] = "00" * 32
        assert semantics.fold(changed) != semantics.PYTHON_SEMANTICS_ID, name


def test_the_procedural_identifier_is_the_native_one_when_the_extension_is_used():
    if na.is_available():
        import libalice
        assert semantics.NATIVE_SEMANTICS_ID == bytes(libalice.semantics_id())
        assert semantics.procedural_semantics_id() == semantics.NATIVE_SEMANTICS_ID
        assert semantics.NATIVE_SEMANTICS_ID != semantics.PYTHON_SEMANTICS_ID
    else:
        assert semantics.NATIVE_SEMANTICS_ID is None
        assert semantics.procedural_semantics_id() == semantics.PYTHON_SEMANTICS_ID


def test_check_accepts_its_own_refuses_another_and_flags_a_missing_one():
    own = semantics.PYTHON_SEMANTICS_ID
    semantics.check(own.hex(), own, "x")
    other = bytes(32).hex()
    with pytest.raises(ValueError) as e:
        semantics.check(other, own, "x")
    assert other in str(e.value) and own.hex() in str(e.value)
    with pytest.warns(semantics.LegacySemanticsWarning):
        semantics.check(None, own, "x")
    with pytest.raises(ValueError, match="strict"):
        semantics.check(None, own, "x", strict=True)
    for bad in ("AB" * 32, "ab" * 31, 7):
        with pytest.raises(ValueError, match="64 lower-case hex"):
            semantics.check(bad, own, "x")


# ------------------------------------------------------------------ procedural payload

def _procedural_file():
    z = ALICEZip()
    data = np.sin(np.linspace(0, 20, 2000)).astype(np.float32)
    blob = z.compress(data)
    hs = 66
    payload = json.loads(blob[hs:])
    assert payload["type"] == "generator", "the input must take the procedural path"
    return z, blob, payload


def _with_payload(blob, payload):
    p = json.dumps(payload, separators=(",", ":")).encode()
    hs = 66
    head = bytearray(blob[:hs])
    head[22:30] = struct.pack("<Q", len(p))
    return bytes(head) + p


def test_the_procedural_payload_records_the_identifier_and_is_compared_on_read():
    z, blob, payload = _procedural_file()
    assert payload["semantics"] == semantics.procedural_semantics_id().hex()
    z.decompress(blob)
    other = dict(payload, semantics=bytes(32).hex())
    with pytest.raises(ValueError) as e:
        z.decompress(_with_payload(blob, other))
    assert bytes(32).hex() in str(e.value)
    assert semantics.procedural_semantics_id().hex() in str(e.value)
    legacy = {k: v for k, v in payload.items() if k != "semantics"}
    with pytest.warns(semantics.LegacySemanticsWarning):
        z.decompress(_with_payload(blob, legacy))
    z.STRICT_SEMANTICS = True
    with pytest.raises(ValueError, match="strict"):
        z.decompress(_with_payload(blob, legacy))


# ------------------------------------------------------------------ media JSON

def _audio(**extra):
    p = {"pattern": "sine_wave", "sample_rate": 8000, "duration": 0.01, "channels": 1,
         "frequencies": [[440.0, 0.5, 0.0]]}
    return json.dumps({"magic": "ALICE_AUD", "params": p, **extra}).encode()


def test_media_json_is_compared_with_the_numpy_identifier():
    g = AudioGenerator()
    own = semantics.PYTHON_SEMANTICS_ID.hex()
    g.decompress(_audio(semantics=own))
    with pytest.raises(ValueError) as e:
        g.decompress(_audio(semantics=bytes(32).hex()))
    assert own in str(e.value)
    with pytest.warns(semantics.LegacySemanticsWarning):
        g.decompress(_audio())
    with pytest.raises(ValueError, match="strict"):
        g.decompress(_audio(), strict=True)


# ------------------------------------------------------------------ residual files

def test_the_residual_writer_records_the_procedural_identifier_with_versions_5_to_7():
    original = np.array([5.0, 5.5, 6.0, 4.0], dtype=np.float32)
    generated = np.array([4.0, 5.0, 6.5, 4.25])
    rd = ResidualCompressor(method=M.NONE).compress_original(original, generated)
    h = header(rd.to_bytes())
    assert (h["version"], h["semantics"]) == (5, semantics.procedural_semantics_id().hex())
    back = ResidualData.from_bytes(rd.to_bytes())
    assert back.semantics == semantics.procedural_semantics_id()
    assert back.to_bytes() == rd.to_bytes()
    rd64 = ResidualCompressor(method=M.NONE).compress_original(original.astype(np.float64), generated)
    assert header(rd64.to_bytes())["version"] == 7


def test_the_files_written_by_each_writer_carry_their_writers_identifier():
    py = (DATA / "python_semantics.bin").read_bytes()
    assert header(py)["semantics"] == semantics.PYTHON_SEMANTICS_ID.hex()
    rust = (DATA / "rust_semantics.bin").read_bytes()
    stored = header(rust)["semantics"]
    assert stored != semantics.PYTHON_SEMANTICS_ID.hex()
    if semantics.NATIVE_SEMANTICS_ID is not None:
        assert stored == semantics.NATIVE_SEMANTICS_ID.hex()
    # each is read where this process regenerates with the same arithmetic and
    # refused naming both identifiers where it does not
    own = semantics.procedural_semantics_id().hex()
    for blob, written in ((py, semantics.PYTHON_SEMANTICS_ID.hex()), (rust, stored)):
        if written == own:
            assert ResidualData.from_bytes(blob, strict=True).semantics.hex() == written
        else:
            with pytest.raises(ValueError) as e:
                ResidualData.from_bytes(blob)
            assert written in str(e.value) and own in str(e.value)


def test_a_residual_without_the_identifier_is_legacy_and_strict_refuses_it():
    legacy = (DATA / "python_none.bin").read_bytes()
    assert "semantics" not in header(legacy)
    with pytest.warns(semantics.LegacySemanticsWarning):
        assert ResidualData.from_bytes(legacy).semantics is None
    with pytest.raises(ValueError, match="strict"):
        ResidualData.from_bytes(legacy, strict=True)


def test_the_identifier_on_versions_1_to_4_is_refused():
    rust = (DATA / "rust_semantics.bin").read_bytes()
    h = header(rust)
    h["version"] = 2
    n = struct.unpack("<I", rust[:4])[0]
    j = json.dumps(h, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="only read with versions 5 to 7"):
        ResidualData.from_bytes(struct.pack("<I", len(j)) + j + rust[4 + n:])
