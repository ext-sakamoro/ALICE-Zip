#!/usr/bin/env python3
"""
ALICE-Zip Command Line Interface
================================

Usage:
    alice-zip compress <input> [-o <output>]
    alice-zip decompress <input> [-o <output>]
    alice-zip image <input> [-o <output>] [--regenerate <path>]
    alice-zip audio <input> [-o <output>] [--regenerate <path>]
    alice-zip video <input> [-o <output>] [--regenerate <path>]
    alice-zip media <input> [-o <output>]
    alice-zip info <file>
    alice-zip benchmark <input>

Author: Moroya Sakamoto
License: MIT
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from .core import ALICEZip, ALICE_VERSION, ALICE_FINGERPRINT


def main():
    parser = argparse.ArgumentParser(
        prog='alice-zip',
        description='ALICE-Zip: Procedural Generation Compression'
    )
    parser.add_argument('--version', action='version',
                        version=f'alice-zip {ALICE_VERSION[0]}.{ALICE_VERSION[1]}')
    parser.add_argument('--debug', action='store_true',
                        help='Show full stack trace on error')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Compress command
    compress_parser = subparsers.add_parser('compress', help='Compress data to .alice format')
    compress_parser.add_argument('input', help='Input file')
    compress_parser.add_argument('-o', '--output', help='Output .alice file')

    # Decompress command
    decompress_parser = subparsers.add_parser('decompress', help='Decompress .alice file')
    decompress_parser.add_argument('input', help='Input .alice file')
    decompress_parser.add_argument('-o', '--output', help='Output file')

    # Image command
    image_parser = subparsers.add_parser('image', help='Compress image (JPEG/PNG/BMP)')
    image_parser.add_argument('input', help='Input image file')
    image_parser.add_argument('-o', '--output', help='Output .alice file')
    image_parser.add_argument('--regenerate', help='Regenerate image to file')

    # Audio command
    audio_parser = subparsers.add_parser('audio', help='Compress audio (WAV)')
    audio_parser.add_argument('input', help='Input audio file')
    audio_parser.add_argument('-o', '--output', help='Output .alice file')
    audio_parser.add_argument('--regenerate', help='Regenerate audio to file')

    # Video command
    video_parser = subparsers.add_parser('video', help='Compress video (MP4/AVI)')
    video_parser.add_argument('input', help='Input video file')
    video_parser.add_argument('-o', '--output', help='Output .alice file')
    video_parser.add_argument('--regenerate', help='Regenerate video to file')

    # Media command (auto-detect)
    media_parser = subparsers.add_parser('media', help='Compress any media (auto-detect)')
    media_parser.add_argument('input', help='Input media file')
    media_parser.add_argument('-o', '--output', help='Output .alice file')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show .alice file information')
    info_parser.add_argument('file', help='.alice file to inspect')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark compression')
    benchmark_parser.add_argument('input', help='Input file to benchmark')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'compress':
            cmd_compress(args)
        elif args.command == 'decompress':
            cmd_decompress(args)
        elif args.command == 'image':
            cmd_image(args)
        elif args.command == 'audio':
            cmd_audio(args)
        elif args.command == 'video':
            cmd_video(args)
        elif args.command == 'media':
            cmd_media(args)
        elif args.command == 'info':
            cmd_info(args)
        elif args.command == 'benchmark':
            cmd_benchmark(args)
    except Exception as e:
        if args.debug:
            raise  # Show full stack trace for debugging
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_compress(args):
    """Compress data to .alice format"""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix('.alice')

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    original_size = input_path.stat().st_size

    zipper = ALICEZip()
    zipper.compress_file(input_path, output_path)

    compressed_size = output_path.stat().st_size

    print(f"Compressed: {input_path}")
    print(f"  Original:   {original_size:,} bytes")
    print(f"  Compressed: {compressed_size:,} bytes")
    print(f"  Ratio:      {original_size/compressed_size:.1f}x")
    print(f"  Output:     {output_path}")


def cmd_decompress(args):
    """Decompress .alice file"""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix('.bin')

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Use decompress_file for proper streaming - it handles everything internally
    zipper = ALICEZip()
    result_path = zipper.decompress_file(input_path, output_path)

    decompressed_size = result_path.stat().st_size

    print(f"Decompressed: {input_path}")
    print(f"  Output: {result_path}")
    print(f"  Size:   {decompressed_size:,} bytes")


def cmd_image(args):
    """Compress/decompress image"""
    try:
        from .media_processors import ImageProcessor
    except ImportError:
        print("Error: Image support requires Pillow. Install with: pip install alice-zip[images]")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    processor = ImageProcessor()

    if args.regenerate:
        # Decompress - pass file path, let processor handle streaming
        image = processor.decompress_file(input_path)

        from PIL import Image
        img = Image.fromarray(image)
        img.save(args.regenerate)
        print(f"Regenerated image: {args.regenerate}")
    else:
        # Compress - pass file path, let processor handle streaming
        output_path = Path(args.output) if args.output else input_path.with_suffix('.alice')
        processor.compress_file(input_path, output_path)

        from PIL import Image
        img = Image.open(input_path)
        original_size = img.size[0] * img.size[1] * len(img.getbands())
        compressed_size = output_path.stat().st_size

        print(f"Compressed image: {input_path}")
        print(f"  Original:   {original_size:,} bytes")
        print(f"  Compressed: {compressed_size:,} bytes")
        print(f"  Ratio:      {original_size/compressed_size:.1f}x")
        print(f"  Output:     {output_path}")


def cmd_audio(args):
    """Compress/decompress audio"""
    try:
        from .media_processors import AudioProcessor
    except ImportError:
        print("Error: Audio support requires SciPy. Install with: pip install alice-zip[audio]")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    processor = AudioProcessor()

    if args.regenerate:
        # Decompress - pass file path, let processor stream to output
        processor.decompress_to_file(input_path, args.regenerate)
        print(f"Regenerated audio: {args.regenerate}")
    else:
        # Compress - pass file path, let processor handle streaming
        output_path = Path(args.output) if args.output else input_path.with_suffix('.alice')
        original_size = processor.compress_file(input_path, output_path)
        compressed_size = output_path.stat().st_size

        print(f"Compressed audio: {input_path}")
        print(f"  Original:   {original_size:,} bytes")
        print(f"  Compressed: {compressed_size:,} bytes")
        print(f"  Ratio:      {original_size/compressed_size:.1f}x")
        print(f"  Output:     {output_path}")


def cmd_video(args):
    """Compress/decompress video"""
    try:
        from .media_processors import VideoProcessor
    except ImportError:
        print("Error: Video support requires OpenCV. Install with: pip install alice-zip[video]")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    processor = VideoProcessor()

    if args.regenerate:
        # Decompress - pass file paths, let processor handle streaming
        processor.decompress_file(input_path, args.regenerate)
        print(f"Regenerated video: {args.regenerate}")
    else:
        # Compress - pass file path, let processor handle streaming
        output_path = Path(args.output) if args.output else input_path.with_suffix('.alice')
        original_size = input_path.stat().st_size
        processor.compress_file(input_path, output_path)
        compressed_size = output_path.stat().st_size

        print(f"Compressed video: {input_path}")
        print(f"  Original:   {original_size:,} bytes")
        print(f"  Compressed: {compressed_size:,} bytes")
        print(f"  Ratio:      {original_size/compressed_size:.1f}x")
        print(f"  Output:     {output_path}")


def cmd_media(args):
    """Auto-detect and compress media"""
    try:
        from .media_processors import MediaCompressor
    except ImportError:
        print("Error: Media support requires optional dependencies. Install with: pip install alice-zip[all]")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix('.alice')
    original_size = input_path.stat().st_size

    # Pass file paths - let compressor handle streaming internally
    compressor = MediaCompressor()
    compressor.compress_file(input_path, output_path)

    compressed_size = output_path.stat().st_size

    print(f"Compressed media: {input_path}")
    print(f"  Original:   {original_size:,} bytes")
    print(f"  Compressed: {compressed_size:,} bytes")
    print(f"  Ratio:      {original_size/compressed_size:.1f}x")
    print(f"  Output:     {output_path}")


def cmd_info(args):
    """Show .alice file information"""
    zipper = ALICEZip()
    info = zipper.info(args.file)

    print(f"ALICE-Zip File: {args.file}")
    print("-" * 50)
    for key, value in info.items():
        if key == 'alice_fingerprint':
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif key == 'metadata':
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


def cmd_benchmark(args):
    """Benchmark compression methods"""
    import time
    import lzma
    import zlib

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    file_size = input_path.stat().st_size

    # Benchmark requires loading data into memory - enforce hard limit
    MAX_BENCHMARK_SIZE = 100 * 1024 * 1024  # 100MB limit for benchmarks
    if file_size > MAX_BENCHMARK_SIZE:
        raise MemoryError(
            f"File too large for benchmark ({file_size / (1024**2):.1f} MB). "
            f"Maximum benchmark file size: {MAX_BENCHMARK_SIZE // (1024**2)} MB. "
            f"Note: Benchmark loads entire file into RAM to test compression algorithms."
        )

    print(f"Loading {file_size / (1024**2):.1f} MB for benchmark...")
    print(f"  Note: Benchmark requires ~{file_size * 2 / (1024**2):.0f} MB RAM (data + working copy)")

    # Read input data using mmap for efficient memory usage
    if input_path.suffix == '.npy':
        # np.load with mmap_mode='r' for memory-mapped read
        data = np.load(input_path, mmap_mode='r')
        # Copy to regular array for benchmark (algorithms need writable data)
        data = np.array(data)
    else:
        import mmap
        with open(input_path, 'rb') as f:
            # Use mmap for memory-efficient file access
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                raw = bytes(mm)  # Copy to bytes for processing
        try:
            data = np.array([float(x) for x in raw.decode().split()])
        except (ValueError, UnicodeDecodeError):
            data = np.frombuffer(raw, dtype=np.uint8)

    original_size = data.nbytes
    results = {'original_size': original_size}

    # Benchmark ALICE-Zip
    zipper = ALICEZip()
    start = time.perf_counter()
    alice_compressed = zipper.compress(data)
    compress_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    alice_decompressed = zipper.decompress(alice_compressed)
    decompress_time = (time.perf_counter() - start) * 1000

    alice_lossless = np.array_equal(data.flatten(), alice_decompressed.flatten())
    results['alice_zip'] = {
        'compressed_size': len(alice_compressed),
        'ratio': original_size / len(alice_compressed) if len(alice_compressed) > 0 else 0,
        'compress_time_ms': compress_time,
        'decompress_time_ms': decompress_time,
        'lossless': alice_lossless
    }

    # Benchmark zlib (ZIP)
    raw_bytes = data.tobytes()
    start = time.perf_counter()
    zlib_compressed = zlib.compress(raw_bytes, level=6)
    compress_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    _ = zlib.decompress(zlib_compressed)  # Result unused, timing only
    decompress_time = (time.perf_counter() - start) * 1000

    results['zip'] = {
        'compressed_size': len(zlib_compressed),
        'ratio': original_size / len(zlib_compressed) if len(zlib_compressed) > 0 else 0,
        'compress_time_ms': compress_time,
        'decompress_time_ms': decompress_time,
        'lossless': True
    }

    # Benchmark LZMA
    start = time.perf_counter()
    lzma_compressed = lzma.compress(raw_bytes, preset=6)
    compress_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    _ = lzma.decompress(lzma_compressed)  # Result unused, timing only
    decompress_time = (time.perf_counter() - start) * 1000

    results['lzma_7zip'] = {
        'compressed_size': len(lzma_compressed),
        'ratio': original_size / len(lzma_compressed) if len(lzma_compressed) > 0 else 0,
        'compress_time_ms': compress_time,
        'decompress_time_ms': decompress_time,
        'lossless': True
    }

    # Determine winner (best ratio)
    best_method = max(['alice_zip', 'zip', 'lzma_7zip'], key=lambda m: results[m]['ratio'])
    results['winner'] = best_method

    print(f"Benchmark: {input_path}")
    print(f"  Original size: {results['original_size']:,} bytes")
    print("-" * 50)

    for method in ['alice_zip', 'zip', 'lzma_7zip']:
        if method in results:
            r = results[method]
            print(f"  {method}:")
            print(f"    Compressed: {r['compressed_size']:,} bytes")
            print(f"    Ratio:      {r['ratio']:.1f}x")
            print(f"    Compress:   {r['compress_time_ms']:.1f} ms")
            print(f"    Decompress: {r['decompress_time_ms']:.1f} ms")
            print(f"    Lossless:   {r['lossless']}")

    print("-" * 50)
    print(f"  Winner: {results['winner']}")


if __name__ == '__main__':
    main()
