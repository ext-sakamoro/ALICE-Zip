"""Decompression is bounded by the length the caller (or a header) states

oracle (from the contract, not read from the code): exactly ``limit`` bytes
pass and one byte less is refused with OutputTooLarge, a bomb is refused
without its expansion being held, and every residual reader refuses a payload
past the length its header implies.
"""

import lzma
import tracemalloc
import zlib

import numpy as np
import pytest

from alice_zip import _bounded
from alice_zip._bounded import OutputTooLarge
from alice_zip.residual_compression import ResidualCompressionMethod as M
from alice_zip.residual_compression import ResidualCompressor

DATA = bytes(range(256)) * 16  # 4096 bytes


def test_lzma_passes_at_the_limit_and_refuses_one_byte_less():
    for fmt in (lzma.FORMAT_XZ, lzma.FORMAT_ALONE):
        packed = lzma.compress(DATA, format=fmt)
        assert _bounded.lzma_decompress(packed, len(DATA)) == DATA
        with pytest.raises(_bounded.OutputTooLarge) as e:
            _bounded.lzma_decompress(packed, len(DATA) - 1)
        assert e.value.limit == len(DATA) - 1


def test_lzma_reads_concatenated_streams_as_lzma_decompress_does_with_the_limit_on_the_sum():
    two = lzma.compress(DATA) + lzma.compress(DATA)
    assert _bounded.lzma_decompress(two, 2 * len(DATA)) == lzma.decompress(two)
    with pytest.raises(_bounded.OutputTooLarge):
        _bounded.lzma_decompress(two, 2 * len(DATA) - 1)
    trailing = lzma.compress(DATA) + b"not a stream"
    assert _bounded.lzma_decompress(trailing, len(DATA)) == lzma.decompress(trailing)


def test_a_truncated_stream_is_an_error_not_a_short_result():
    packed = lzma.compress(DATA)
    with pytest.raises(lzma.LZMAError):
        _bounded.lzma_decompress(packed[:-8], len(DATA))
    z = zlib.compress(DATA)
    with pytest.raises(zlib.error):
        _bounded.zlib_decompress(z[:-4], len(DATA))


def test_zlib_passes_at_the_limit_and_refuses_one_byte_less():
    raw = zlib.compressobj(6, zlib.DEFLATED, -15)
    raw_deflate = raw.compress(DATA) + raw.flush()
    for wbits, packed in ((zlib.MAX_WBITS, zlib.compress(DATA)), (-15, raw_deflate)):
        assert _bounded.zlib_decompress(packed, len(DATA), wbits) == DATA
        with pytest.raises(_bounded.OutputTooLarge):
            _bounded.zlib_decompress(packed, len(DATA) - 1, wbits)


def test_a_bomb_is_refused_without_holding_its_expansion():
    bomb_size = 64 * 1024 * 1024
    for packed, call in ((lzma.compress(bytes(bomb_size)), _bounded.lzma_decompress),
                         (zlib.compress(bytes(bomb_size), 9), _bounded.zlib_decompress)):
        tracemalloc.start()
        with pytest.raises(_bounded.OutputTooLarge):
            call(packed, 1024)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        # the decoder's own dictionary (8 MiB at preset 6) is traced too; the expansion is 64 MiB
        assert peak < 16 * 1024 * 1024, peak


def test_a_negative_or_non_integer_limit_is_refused():
    for bad in (-1, 1.5, True, None):
        with pytest.raises(ValueError):
            _bounded.lzma_decompress(lzma.compress(b"x"), bad)


def test_the_zstd_frame_content_size_is_read_from_the_header():
    # RFC 8878 3.1.1.1: magic, descriptor, then the size (single segment, 1 / 2 / 4 / 8 bytes)
    magic = (0xFD2FB528).to_bytes(4, "little")
    assert _bounded.zstd_content_size(magic + bytes([0x20, 200])) == 200
    assert _bounded.zstd_content_size(magic + bytes([0x60]) + (1000 - 256).to_bytes(2, "little")) == 1000
    assert _bounded.zstd_content_size(magic + bytes([0xA0]) + (70000).to_bytes(4, "little")) == 70000
    # not single segment, no size field: a window byte follows and no size is stated
    with pytest.raises(ValueError):
        _bounded.zstd_content_size(magic + bytes([0x00, 0x50]))
    with pytest.raises(ValueError):
        _bounded.zstd_content_size(b"\x00" * 8)


@pytest.mark.parametrize("method,bits", [(M.LZMA, None), (M.ZLIB, None), (M.BITDELTA, None),
                                         (M.ZSTD, None), (M.QUANTIZED, 8), (M.QUANTIZED, 16)])
def test_a_residual_past_the_length_its_header_states_is_refused(method, bits):
    data = np.full(4096, 0.5, dtype=np.float32)
    rc = ResidualCompressor(method=method, quantization_bits=bits)
    rd = rc.compress_residual(data)
    assert rd.method == method, "fell back to another encoding"
    assert rc.decompress_residual(rd).size == data.size
    rd.original_shape = (data.size - 1,)
    with pytest.raises(_bounded.OutputTooLarge):
        rc.decompress_residual(rd)


def test_the_lzma_fallback_of_the_generators_is_bounded_by_shape_and_dtype():
    from alice_zip.generators import decompress_from_lzma
    values = np.arange(16, dtype=np.float32)
    packed = lzma.compress(values.tobytes())
    assert (decompress_from_lzma(packed, (16,), "float32") == values).all()
    with pytest.raises(_bounded.OutputTooLarge):
        decompress_from_lzma(packed, (15,), "float32")


@pytest.mark.parametrize("method", [M.LZMA, M.ZLIB, M.NONE])
def test_a_version_4_payload_past_its_stated_length_is_refused(method):
    from alice_zip.residual_compression import ResidualData
    n = 4096
    original = np.full(n, 2.5, dtype=np.float32)
    # every 512th generated value is far from the original, kept as an exception (a version 4 file)
    generated = np.where(np.arange(n) % 512 == 0, 1e10, 2.0)
    rd = ResidualCompressor(method=method).compress_original(original, generated)
    assert rd.layout == 4
    assert ResidualData.from_bytes(rd.to_bytes()).original_shape == (n,)
    rd.original_shape = (n - 1,)
    expected = OutputTooLarge if method != M.NONE else ValueError
    with pytest.raises(expected):
        ResidualData.from_bytes(rd.to_bytes())


def _rust_container(codes, bits, codec):
    import struct
    if codec == 0:
        c = zlib.compressobj(6, zlib.DEFLATED, -15)
        payload = c.compress(codes) + c.flush()
    else:
        payload = lzma.compress(codes, format=lzma.FORMAT_ALONE)
    return (b"\xfc" + bytes([codec, bits]) + struct.pack("<dd", 0.0, 1.0)
            + struct.pack("<I", len(payload)) + payload)


@pytest.mark.parametrize("bits,codec", [(8, 0), (8, 1), (16, 0), (16, 1)])
def test_the_rust_quantized_container_is_bounded_by_value_count_and_bit_width(bits, codec):
    from alice_zip.residual_compression import _decode_rust_quantized_container
    n = 1024
    codes = bytes(n * bits // 8)
    data = _rust_container(codes, bits, codec)
    assert _decode_rust_quantized_container(data, n).size == n
    with pytest.raises(OutputTooLarge):
        _decode_rust_quantized_container(data, n - 1)


def test_the_earlier_float_delta_and_its_differences_are_bounded_by_the_shape():
    from alice_zip.residual_compression import ResidualData, decompress_delta_differences
    deltas = np.full(256, 0.25, dtype="<f4")
    rd = ResidualData(method=M.DELTA, compressed_data=lzma.compress(deltas.tobytes()),
                      original_shape=(256,), original_dtype="float32",
                      compression_ratio=1.0, base_value=0.0)
    assert ResidualCompressor().decompress_residual(rd).size == 256
    assert decompress_delta_differences(rd).size == 256
    rd.original_shape = (255,)
    with pytest.raises(OutputTooLarge):
        ResidualCompressor().decompress_residual(rd)
    with pytest.raises(OutputTooLarge):
        decompress_delta_differences(rd)


@pytest.mark.parametrize("bits,width", [(8, 1), (16, 2), (None, 4)])
def test_the_earlier_residual_path_of_the_designer_is_bounded_by_the_shape(bits, width):
    from alice_zip.analyzers import ProceduralCompressionDesigner
    from alice_zip.generators import CompressionEngine
    signal = np.sin(np.linspace(0, 20, 512)).astype(np.float32)
    designer = ProceduralCompressionDesigner()
    result = designer.compress(signal, enable_lossless=False)
    if result.engine_used != CompressionEngine.PROCEDURAL:
        pytest.skip("the signal was not fitted procedurally")
    result.residual_data = lzma.compress(bytes(512 * width))
    result.metadata.pop("residual_format", None)
    result.metadata["residual_quantized"] = bits
    result.metadata["residual_min"], result.metadata["residual_max"] = 0.0, 1.0
    result.metadata["shape"] = [512]
    assert designer.decompress(result, allow_approximate=True).size == 512
    # one value more than the shape holds
    result.residual_data = lzma.compress(bytes(513 * width))
    with pytest.raises(OutputTooLarge):
        designer.decompress(result, allow_approximate=True)
