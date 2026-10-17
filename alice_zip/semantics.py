"""Identifiers of the arithmetic a stored value was regenerated with.

A file that stores parameters (or a residual against values generated from
parameters) is only meaningful together with the arithmetic that turns the
parameters back into values. Every such file this package writes records that
arithmetic as ``"semantics"`` (64 hex digits), and a reader compares the
**stored** identifier, not its own build's:

* the identifier of the arithmetic this process regenerates with: decoded;
* another identifier: ``ValueError`` naming both;
* no identifier (written before 3.0.0): decoded with a
  :class:`LegacySemanticsWarning`, or refused when the caller asks for a
  strict read.

There are two arithmetics, because the procedural generators run either in the
native extension (``libalice``, the Rust crate ``alice-zip``) or in the pure
Python fallback (numpy):

* :data:`NATIVE_SEMANTICS_ID` is ``alice_zip::law::SEMANTICS_ID`` of the Rust
  crate the extension was built from (``None`` without the extension);
* :data:`PYTHON_SEMANTICS_ID` is this package's own: the SHA-256 fold of
  :data:`PYTHON_SEMANTICS_PINS` (each name, sorted, as a big-endian u32 length
  and its bytes, then the 32-byte digest), the same fold the Rust crate uses.
  The pins are the digests of the pure Python generators' output for fixed
  inputs (``tests/test_semantics.py`` recomputes each one) and of the names of
  the generators the identifier covers. It changes when a pinned output
  changes, not with the package version.

:func:`procedural_semantics_id` is the one the procedural generators use in
this process; the media generators (``media_generators``) always run on numpy
and use :data:`PYTHON_SEMANTICS_ID`.
"""
import hashlib
import struct
import warnings
from typing import Optional

from . import native_accelerator as _accel

# name -> SHA-256 (hex). The output pins are the little-endian float32 bytes of:
#   sine:       _python_sine_wave(4096, 7.0, 1.5, 0.25, -0.5)
#   fourier:    _python_fourier_generate(4096, [(3, 1.0, 0.1), (17, 0.5, 1.2), (200, 0.25, -0.7)], 0.125)
#   polynomial: _python_polynomial_generate(4096, [0.5, -1.25, 2.0, 0.75])
#   perlin:     _python_perlin_advanced(64, 48, 1234, 8.0, 4, 0.5, 2.0)
#   white noise: AudioGenerator().generate(seed 2024), int16 (tests/test_media_generators.py)
PYTHON_SEMANTICS_PINS = {
    "golden python sine_wave": "d54b3bf0f51e2d99c200eec59a2c46a7511e0f475f8e962c1471f5f847486f35",
    "golden python fourier_generate": "9cf69b6b36a9312a43d44214287e42623b16df92e0271916cf4b94915db9890f",
    "golden python polynomial_generate": "13e63cf7b504260fe532941a0d1424d2ca0f8f7a98f06cad22c8aba4d727cc89",
    "golden python perlin_advanced": "6690225bdf4d51af09d042b5c6c8c42ca37483cdde464dcf0c5580a91f08bd90",
    "golden python white_noise": "7f968530a1d13f5724f8ec95625caba4501c5be28d07179d71a788f2641bc6e6",
    "tag covered generators": hashlib.sha256(
        b"sine_wave fourier_generate polynomial_generate perlin_advanced "
        b"media image audio video numpy RandomState"
    ).hexdigest(),
}


def fold(pins: dict) -> bytes:
    """The SHA-256 fold of ``pins`` (name -> hex digest), names sorted"""
    h = hashlib.sha256()
    for name in sorted(pins):
        raw = name.encode("utf-8")
        h.update(struct.pack(">I", len(raw)))
        h.update(raw)
        h.update(bytes.fromhex(pins[name]))
    return h.digest()


PYTHON_SEMANTICS_ID = fold(PYTHON_SEMANTICS_PINS)
NATIVE_SEMANTICS_ID: Optional[bytes] = _accel.native_semantics_id()


def procedural_semantics_id() -> bytes:
    """The identifier of the arithmetic the procedural generators use here"""
    return NATIVE_SEMANTICS_ID if NATIVE_SEMANTICS_ID is not None else PYTHON_SEMANTICS_ID


class LegacySemanticsWarning(UserWarning):
    """A file written before files recorded their arithmetic identifier"""


def check(stored: Optional[str], own: bytes, what: str, strict: bool = False) -> None:
    """Compare the identifier stored in a file with ``own``.

    ``stored`` is the ``"semantics"`` value (64 hex digits) or ``None`` when the
    file has none. Raises ``ValueError`` naming both identifiers on a
    mismatch, and on a missing identifier when ``strict``; otherwise warns
    with :class:`LegacySemanticsWarning` for a missing one.
    """
    if stored is None:
        if strict:
            raise ValueError(
                f"{what}: written before files recorded their arithmetic "
                f"identifier, refused by a strict read")
        warnings.warn(
            f"{what}: written before files recorded their arithmetic identifier; "
            f"decoded with this process's arithmetic ({own.hex()})",
            LegacySemanticsWarning, stacklevel=3)
        return
    if not isinstance(stored, str) or len(stored) != 64 or \
            any(c not in "0123456789abcdef" for c in stored):
        raise ValueError(f"{what}: semantics must be 64 lower-case hex digits, got {stored!r}")
    if stored != own.hex():
        raise ValueError(
            f"{what}: regenerated under semantics {stored}, this process "
            f"regenerates under {own.hex()}")
