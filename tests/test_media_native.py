"""New images come from the Rust crate through the extension; a file reads what it recorded.

The digests of RECORDED were computed by the package at 7f80729, the last with the numpy formulas as the only
image code, on the same files.
"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from alice_zip import media_generators, native_accelerator, semantics
from alice_zip.core import ALICEZip, AlicePayloadType
from alice_zip.media_generators import (AudioGenerator, AudioParams, AudioPattern, ImageGenerator, ImageParams,
                                        ImagePattern, VideoGenerator)

NATIVE = native_accelerator.is_available()
needs_native = pytest.mark.skipif(not NATIVE, reason="needs the native extension")
needs_no_native = pytest.mark.skipif(NATIVE, reason="the case without the native extension")

RECORDED = {
    "solid_color": "d36cca5380d6cd08a239d5b9e29f8d5c3b28c6dc9b0fde1aa46f53027d961f93",
    "gradient_linear": "cf1132d3bfbfa90a4a533159971262d42cbdc2c59075fc2c0adfc1fbeb66379e",
    "gradient_radial": "ef6dce2c0f53073cc17a0871c835534ef2c85097dcd5c52cc75fa53c6f84887d",
    "checkerboard": "29d1d53424ac185cf809f56fff47d09ae5473fcfbb0d7ee9b459ac6404876100",
    "stripes": "010288e30a19edc6f93414d3146158dbdb15d40099862bb6de9d43966527e4c9",
    "noise_perlin": "010288e30a19edc6f93414d3146158dbdb15d40099862bb6de9d43966527e4c9",
    "noise_gaussian": "010288e30a19edc6f93414d3146158dbdb15d40099862bb6de9d43966527e4c9",
    "fractal": "010288e30a19edc6f93414d3146158dbdb15d40099862bb6de9d43966527e4c9",
}
BASE = dict(width=33, height=21, channels=3, color=[10, 200, 40], gradient_start=[10, 200, 40],
            gradient_end=[250, 0, 90], gradient_angle=33.0, gradient_center=[0.25, 0.75], pattern_scale=4.0)


def file_of(pattern, stored, **extra):
    head = {"magic": "ALICE_IMG", "params": dict(BASE, pattern=pattern, **extra)}
    if stored is not None:
        head["semantics"] = stored
    return json.dumps(head).encode()


def digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def params(pattern, **kw):
    base = dict(width=96, height=64, channels=3, gradient_start=(12, 200, 77), gradient_end=(250, 3, 140),
                color=(12, 200, 77), gradient_angle=33.0, pattern_scale=7.0, pattern_seed=7)
    base.update(kw)
    return ImageParams(pattern=pattern, **base)


@pytest.mark.parametrize("pattern", sorted(RECORDED))
def test_a_file_of_the_numpy_identifier_reads_what_it_recorded(pattern):
    data = file_of(pattern, semantics.PYTHON_SEMANTICS_ID.hex())
    assert digest(ImageGenerator().decompress(data)) == RECORDED[pattern]


def test_a_file_with_no_identifier_reads_what_it_recorded_unless_strict():
    data = file_of("gradient_radial", None)
    with pytest.warns(semantics.LegacySemanticsWarning):
        assert digest(ImageGenerator().decompress(data)) == RECORDED["gradient_radial"]
    with pytest.raises(ValueError, match="strict"):
        ImageGenerator().decompress(data, strict=True)


def test_a_file_of_another_identifier_is_refused():
    with pytest.raises(ValueError, match="regenerated under"):
        ImageGenerator().decompress(file_of("fractal", bytes(32).hex()))


@needs_native
def test_a_file_of_the_native_identifier_regenerates_with_the_rust_crate():
    data = file_of("fractal", semantics.NATIVE_SEMANTICS_ID.hex(), pattern_seed=5)
    got = ImageGenerator().decompress(data)
    assert digest(got) != RECORDED["fractal"]
    other = ImageGenerator().decompress(file_of("fractal", semantics.NATIVE_SEMANTICS_ID.hex(), pattern_seed=6))
    assert digest(got) != digest(other)


@needs_native
@pytest.mark.parametrize("pattern,golden", [
    (ImagePattern.SOLID_COLOR, "media_solid_is_the_recorded_bits"),
    (ImagePattern.GRADIENT_LINEAR, "media_linear_gradient_is_the_recorded_bits"),
    (ImagePattern.GRADIENT_RADIAL, "media_radial_gradient_is_the_recorded_bits"),
    (ImagePattern.CHECKERBOARD, "media_checkerboard_is_the_recorded_bits"),
    (ImagePattern.NOISE_PERLIN, "media_noise_is_the_recorded_bits"),
    (ImagePattern.FRACTAL, "media_fractal_is_the_recorded_bits"),
])
def test_python_gets_the_bits_of_the_rust_golden(pattern, golden):
    # the digest entered into alice_zip::law::SEMANTICS_ID (src/semantics.rs); read there, not copied
    from pathlib import Path
    source = (Path(__file__).parent.parent / "src" / "semantics.rs").read_text(encoding="utf-8")
    at = source.index(f'"golden {golden}"')
    want = source[at:].split('hex32("', 1)[1].split('"', 1)[0]
    assert digest(ImageGenerator().generate(params(pattern))) == want


@needs_native
def test_noise_takes_the_seed_and_the_channels():
    a = ImageGenerator().generate(params(ImagePattern.NOISE_PERLIN, pattern_seed=1))
    b = ImageGenerator().generate(params(ImagePattern.NOISE_PERLIN, pattern_seed=2))
    assert a.shape == (64, 96, 3)
    assert (a != b).mean() > 0.5
    one = ImageGenerator().generate(params(ImagePattern.FRACTAL, channels=1))
    four = ImageGenerator().generate(params(ImagePattern.FRACTAL, channels=4))
    assert one.shape == (64, 96, 1)
    assert four.shape == (64, 96, 4)
    assert np.array_equal(one[:, :, 0], four[:, :, 0])
    assert (four[:, :, 3] > 127).any()


@needs_native
def test_a_radial_gradient_has_its_centre_where_the_params_say():
    g = ImageGenerator()
    image = g.generate(params(ImagePattern.GRADIENT_RADIAL, width=33, height=33, gradient_center=(0.5, 0.5)))
    assert tuple(image[16, 16]) == (12, 200, 77)
    for y, x in ((0, 0), (0, 32), (32, 0), (32, 32)):
        assert tuple(image[y, x]) == (250, 3, 140)
    corner = g.generate(params(ImagePattern.GRADIENT_RADIAL, width=9, height=9, gradient_center=(0.0, 0.0)))
    assert tuple(corner[0, 0]) == (12, 200, 77)
    assert tuple(corner[8, 8]) == (250, 3, 140)


@needs_native
@pytest.mark.parametrize("centre", [(0.75, 0.25), (0.25, 0.75)])
def test_a_radial_gradient_off_centre_measures_from_its_farthest_corner(centre):
    # oracle: (12, 200, 77) to (250, 3, 140) at distance / distance to the farthest corner, to one level
    w, h = 17, 9
    image = ImageGenerator().generate(params(ImagePattern.GRADIENT_RADIAL, width=w, height=h, gradient_center=centre))
    cx, cy = centre[0] * (w - 1), centre[1] * (h - 1)
    far = np.hypot(max(cx, w - 1 - cx), max(cy, h - 1 - cy))
    yy, xx = np.mgrid[:h, :w]
    t = (np.hypot(xx - cx, yy - cy) / far)[..., None]
    want = np.array([12, 200, 77]) + (np.array([250, 3, 140]) - np.array([12, 200, 77])) * t
    assert np.abs(image.astype(float) - want).max() <= 1.0


@needs_native
@pytest.mark.parametrize("stamp", [lambda: semantics.NATIVE_SEMANTICS_ID.hex(), lambda: None])
def test_a_file_stating_a_huge_image_is_refused_before_allocating(stamp):
    # the native case would end the process if it reached the allocation
    file = json.dumps({"magic": "ALICE_IMG", "semantics": stamp(), "params": {
        "pattern": "solid_color", "width": 2 ** 45, "height": 1, "channels": 3}}).encode()
    with pytest.raises(MemoryError, match="exceeds the limit"):
        ImageGenerator().decompress(file)


def test_the_size_limit_is_the_limit_it_is_given():
    file = lambda w: json.dumps({"magic": "ALICE_IMG", "semantics": None, "params": {
        "pattern": "solid_color", "width": w, "height": 1, "channels": 1, "color": [9]}}).encode()
    with pytest.warns(Warning):
        assert ImageGenerator().decompress(file(8), max_output_size=8).shape == (1, 8, 1)
    with pytest.raises(MemoryError):
        ImageGenerator().decompress(file(9), max_output_size=8)


def test_the_zipper_passes_its_limit_to_the_media_readers():
    zipper = ALICEZip()
    zipper.MAX_OUTPUT_SIZE = 8
    file = json.dumps({"magic": "ALICE_IMG", "params": {
        "pattern": "solid_color", "width": 9, "height": 1, "channels": 1}}).encode()
    with pytest.raises(MemoryError):
        zipper._decompress_media_json(file, None)


@pytest.mark.parametrize("payload_type,magic,extra", [
    (AlicePayloadType.MEDIA_IMAGE, "ALICE_IMG", {"params": {"pattern": "solid_color", "width": 9, "height": 1, "channels": 1}}),
    (AlicePayloadType.MEDIA_VIDEO, "ALICE_VID", {"keyframes": [[0, {"pattern": "solid_color", "width": 9, "height": 1, "channels": 1}]]})])
def test_the_zipper_passes_its_limit_when_the_header_names_the_payload(monkeypatch, payload_type, magic, extra):
    monkeypatch.setattr(media_generators, "HAS_CV2", True)  # no frame is written
    zipper = ALICEZip()
    zipper.MAX_OUTPUT_SIZE = 8
    with pytest.raises(MemoryError):
        zipper._decode_payload(SimpleNamespace(payload_type=payload_type),
                               json.dumps({"magic": magic, **extra}).encode(), "unused.mp4")


def _audio_file(duration, sample_rate=8, channels=1):
    return json.dumps({"magic": "ALICE_AUD", "semantics": None, "params": {
        "pattern": "silence", "sample_rate": sample_rate, "duration": duration, "channels": channels}}).encode()


@pytest.mark.parametrize("duration", [1e12, 1e300, float("inf")])
def test_an_audio_file_stating_a_huge_duration_is_refused_before_generating(monkeypatch, duration):
    def not_reached(*_args, **_kwargs):  # without the bound this would run for hours
        raise AssertionError("samples were generated")
    monkeypatch.setattr(AudioGenerator, "generate", not_reached)
    with pytest.raises(MemoryError, match="exceeds the limit"):
        AudioGenerator().decompress(_audio_file(duration, 44100, 2))


def test_the_audio_limit_counts_the_bytes_of_the_samples_it_would_return():
    # 8 samples x 2 channels x 2 bytes = 32
    with pytest.warns(Warning):
        samples, rate = AudioGenerator().decompress(_audio_file(1, channels=2), max_output_size=32)
    assert (samples.shape, samples.dtype, rate) == ((8, 2), np.int16, 8)
    assert samples.nbytes == 32
    with pytest.raises(MemoryError):
        AudioGenerator().decompress(_audio_file(1, channels=2), max_output_size=31)
    # a duration the sample rate cuts down to 8 samples is 8 samples
    with pytest.warns(Warning):
        assert AudioGenerator().decompress(_audio_file(1.1), max_output_size=16)[0].nbytes == 16


def test_the_zipper_passes_its_limit_to_the_audio_reader():
    zipper = ALICEZip()
    zipper.MAX_OUTPUT_SIZE = 8
    with pytest.raises(MemoryError):
        zipper._decompress_media_json(_audio_file(1), None)
    with pytest.raises(MemoryError):
        zipper._decode_payload(SimpleNamespace(payload_type=AlicePayloadType.MEDIA_AUDIO), _audio_file(1), None)


def test_a_video_stating_a_huge_keyframe_is_refused(monkeypatch):
    monkeypatch.setattr(media_generators, "HAS_CV2", True)  # no frame is written
    file = json.dumps({"magic": "ALICE_VID", "pattern": "keyframes", "width": 1, "height": 1, "fps": 1, "duration": 1,
                       "keyframes": [[0, {"pattern": "solid_color", "width": 2 ** 45, "height": 1, "channels": 3}]]})
    with pytest.raises(MemoryError):
        VideoGenerator().decompress(file.encode(), "unused.mp4")


@needs_native
def test_the_native_call_refuses_a_huge_image_itself():
    with pytest.raises(ValueError, match="exceeds"):
        native_accelerator.media_image("solid_color", 2 ** 45, 1, (0,), (0,), 0.0, (0.5, 0.5), 8, 42)


@needs_native
@pytest.mark.parametrize("pattern", [ImagePattern.PERIODIC, ImagePattern.UNKNOWN])
def test_the_native_identifier_does_not_cover_what_numpy_computes(pattern):
    file = json.dumps({"magic": "ALICE_IMG", "semantics": semantics.NATIVE_SEMANTICS_ID.hex(), "params": {
        "pattern": pattern.value, "width": 4, "height": 4, "channels": 3}}).encode()
    with pytest.raises(ValueError, match="does not cover"):
        ImageGenerator().decompress(file)


@needs_native
@pytest.mark.parametrize("change", [{"pattern_scale": 0}, {"pattern_scale": -3}, {"pattern_scale": float("nan")},
                                    {"gradient_angle": float("inf")}, {"gradient_angle": 1e300},
                                    {"gradient_center": (float("nan"), 0.5)}])
def test_values_that_name_no_law_are_refused(change):
    with pytest.raises(ValueError):
        ImageGenerator().generate(params(ImagePattern.CHECKERBOARD, **change))


@needs_native
def test_a_keyframe_is_cached_under_its_law_and_its_centre(monkeypatch):
    monkeypatch.setattr(media_generators, "HAS_CV2", True)  # no frame is written
    g = VideoGenerator()
    key = params(ImagePattern.GRADIENT_RADIAL, width=9, height=9, gradient_center=(0.25, 0.25))
    # oracle: each read equals the frame a generator with an empty cache makes
    for first, second in ((True, False), (False, True)):
        g._frame_cache.clear()
        g._get_base_frame(key, 9, 9, recorded=first)
        frame = g._get_base_frame(key, 9, 9, recorded=second)
        assert np.array_equal(frame, VideoGenerator()._get_base_frame(key, 9, 9, recorded=second))
    other = params(ImagePattern.GRADIENT_RADIAL, width=9, height=9, gradient_center=(0.75, 0.75))
    assert not np.array_equal(g._get_base_frame(key, 9, 9), g._get_base_frame(other, 9, 9))


@needs_no_native
@pytest.mark.parametrize("pattern", [ImagePattern.SOLID_COLOR, ImagePattern.GRADIENT_LINEAR,
                                     ImagePattern.GRADIENT_RADIAL, ImagePattern.CHECKERBOARD,
                                     ImagePattern.NOISE_PERLIN, ImagePattern.FRACTAL])
def test_without_the_extension_a_new_image_is_refused(pattern):
    with pytest.raises(RuntimeError, match="native extension"):
        ImageGenerator().generate(params(pattern))


@pytest.mark.parametrize("pattern", [ImagePattern.STRIPES, ImagePattern.NOISE_GAUSSIAN])
def test_an_image_pattern_with_no_law_raises(pattern):
    with pytest.raises(ValueError, match="no law"):
        ImageGenerator().generate(params(pattern))


@pytest.mark.parametrize("pattern", [AudioPattern.NOISE_PINK, AudioPattern.SPEECH, AudioPattern.MUSIC,
                                     AudioPattern.UNKNOWN])
def test_an_audio_pattern_with_no_law_raises_and_its_file_reads_what_it_recorded(pattern):
    p = AudioParams(pattern=pattern, sample_rate=8000, duration=0.1)
    with pytest.raises(ValueError, match="no law"):
        AudioGenerator().generate(p)
    data = json.dumps({"magic": "ALICE_AUD", "semantics": semantics.PYTHON_SEMANTICS_ID.hex(),
                       "params": {"pattern": pattern.value, "sample_rate": 8000, "duration": 0.1,
                                  "channels": 1}}).encode()
    samples, rate = AudioGenerator().decompress(data)
    assert rate == 8000
    assert len(samples) > 0
    assert not np.any(samples)
