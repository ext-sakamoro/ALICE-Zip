"""Every function of the Python package whose name says it writes, and how it
identifies the meaning of what it writes (ALICE-Law G8).

The table is count-locked against the source (read with ast): a function named
like a writer (compress / encode / to_bytes / write / save / serial / store /
dump / export / pack, not decompress / decode) that is not in WRITERS fails,
and so does an entry the source no longer has. The decisions that put an
identifier in a file name the test of tests/test_semantics.py that checks it,
and that test must exist.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "alice_zip"

WORDS = ("compress", "encode", "to_bytes", "write", "save", "serial", "store", "dump",
         "export", "pack")
READS = ("decompress", "decode")

# (file, function) -> (decision, reason or the test that checks the identifier)
WRITERS = {
    ("core.py", "to_bytes"): ("header", "AliceFileHeader: the fixed 66-byte ALICE_ZIP header labels the payload; the identity is in the payload"),
    ("core.py", "compress"): ("semantics", "test_the_procedural_payload_records_the_identifier_and_is_compared_on_read"),
    ("core.py", "_serialize_procedural_params"): ("semantics", "test_the_procedural_payload_records_the_identifier_and_is_compared_on_read"),
    ("core.py", "_serialize_lzma_payload"): ("literal", "the LZMA fallback stores the input bytes; decoding regenerates nothing and the header's original_hash is checked"),
    ("core.py", "compress_file"): ("semantics", "test_the_procedural_payload_records_the_identifier_and_is_compared_on_read"),
    ("cli.py", "cmd_compress"): ("semantics", "test_the_procedural_payload_records_the_identifier_and_is_compared_on_read"),
    ("residual_compression.py", "to_bytes"): ("semantics", "test_the_residual_writer_records_the_procedural_identifier_with_versions_5_to_7"),
    ("residual_compression.py", "compress_residual"): ("semantics", "test_the_residual_writer_records_the_procedural_identifier_with_versions_5_to_7"),
    ("residual_compression.py", "compress_original"): ("semantics", "test_the_residual_writer_records_the_procedural_identifier_with_versions_5_to_7"),
    ("residual_compression.py", "_compress_delta"): ("literal", "encodes the caller's residual bytes inside ResidualData.to_bytes"),
    ("residual_compression.py", "_compress_zstd"): ("literal", "compresses the caller's residual bytes inside ResidualData.to_bytes"),
    ("residual_compression.py", "_bit_delta_encode"): ("literal", "bit pattern deltas of the caller's values"),
    ("residual_compression.py", "_bit_delta_encode32"): ("literal", "bit pattern deltas of the caller's values"),
    ("residual_compression.py", "estimate_total_compression"): ("not-durable", "returns an estimate, writes nothing"),
    ("analyzers.py", "compress"): ("not-durable", "returns an in-memory CompressionResult; ALICEZip.compress writes it"),
    ("analyzers.py", "compress_with_lzma"): ("not-durable", "returns an in-memory CompressionResult"),
    ("generators.py", "total_compressed_size"): ("not-durable", "a size, writes nothing"),
    ("media_generators.py", "encode_coefficients_b64"): ("literal", "base64 of the caller's coefficients"),
    ("core.py", "_writer_dtype"): ("not-durable", "validates a dtype name read from a header, writes nothing"),
}
DECISIONS = {"header", "semantics", "literal", "not-durable"}


def writers_in_source():
    found = set()
    files = sorted(PKG.glob("*.py"))
    assert len(files) > 5, f"read too few files: {len(files)}"
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if any(w in name for w in WORDS) and not any(r in name for r in READS):
                    found.add((path.name, name))
    return found


def test_the_writer_table_is_every_writer_in_the_source():
    found = writers_in_source()
    listed = set(WRITERS)
    assert found - listed == set(), f"writers without a decision: {sorted(found - listed)}"
    assert listed - found == set(), f"listed but not in the source: {sorted(listed - found)}"
    assert len(found) == 19


def test_every_decision_is_known_and_every_named_test_exists():
    tests = (ROOT / "tests" / "test_semantics.py").read_text(encoding="utf-8")
    for key, (decision, why) in WRITERS.items():
        assert decision in DECISIONS, key
        if decision == "semantics":
            assert f"def {why}(" in tests, (key, why)
        else:
            assert len(why) > 12, key
