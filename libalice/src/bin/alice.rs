//! ALICE-Zip CLI
//!
//! Command-line interface for ALICE-Zip procedural compression.
//!
//! Usage:
//!   alice compress <input> -o <output> [--quality <lossy|near-lossless|lossless>]
//!   alice decompress <input> -o <output>
//!   alice info <file>
//!   alice benchmark <input>

use clap::{Parser, Subcommand, ValueEnum};
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::PathBuf;
use std::time::Instant;

// Use the library's compression module (DRY principle)
use alice_core::compression::{
    zlib_compress, zlib_decompress,
    quantize_8bit, dequantize_8bit,
};

/// ALICE-Zip: Procedural Compression for Scientific Data
#[derive(Parser)]
#[command(name = "alice")]
#[command(author = "Moroya Sakamoto")]
#[command(version = "0.1.0")]
#[command(about = "High-performance procedural compression for scientific data", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a file
    Compress {
        /// Input file path
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Compression quality
        #[arg(short, long, value_enum, default_value = "near-lossless")]
        quality: Quality,

        /// Quantization bits (for near-lossless mode)
        #[arg(long, default_value = "8")]
        bits: u8,

        /// Compression level (1-9, higher = better compression but slower)
        #[arg(short = 'L', long, default_value = "6", value_parser = clap::value_parser!(u32).range(1..=9))]
        level: u32,

        /// Force raw LZMA (skip procedural analysis)
        #[arg(long)]
        raw: bool,
    },

    /// Decompress a file
    Decompress {
        /// Input .alz file path
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Show file information
    Info {
        /// File path (.alz or raw data)
        file: PathBuf,
    },

    /// Benchmark compression
    Benchmark {
        /// Input file path
        input: PathBuf,

        /// Number of iterations
        #[arg(short, long, default_value = "3")]
        iterations: u32,

        /// Compression level (1-9, higher = better compression but slower)
        #[arg(short = 'L', long, default_value = "6", value_parser = clap::value_parser!(u32).range(1..=9))]
        level: u32,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Quality {
    /// Maximum compression, some quality loss
    Lossy,
    /// Good compression with minimal quality loss (8-bit quantization)
    #[value(name = "near-lossless")]
    NearLossless,
    /// Perfect reconstruction
    Lossless,
}

// ============================================================================
// .alz File Format
// ============================================================================

/// ALICE-Zip file magic number: "ALZ\x01"
const MAGIC: [u8; 4] = [0x41, 0x4C, 0x5A, 0x01];

/// File format version
const FORMAT_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum CompressionMode {
    RawLzma = 0,        // Pure LZMA, no procedural
    Polynomial = 1,     // Polynomial fit + residual
    Fourier = 2,        // Fourier fit + residual
    Perlin = 3,         // Perlin noise + residual
    Quantized8 = 10,    // 8-bit quantized residual
    Quantized16 = 11,   // 16-bit quantized residual
    Lossless = 20,      // Full precision residual
}

impl From<u8> for CompressionMode {
    fn from(v: u8) -> Self {
        match v {
            0 => CompressionMode::RawLzma,
            1 => CompressionMode::Polynomial,
            2 => CompressionMode::Fourier,
            3 => CompressionMode::Perlin,
            10 => CompressionMode::Quantized8,
            11 => CompressionMode::Quantized16,
            20 => CompressionMode::Lossless,
            _ => CompressionMode::RawLzma,
        }
    }
}

/// .alz file header (fixed size: 32 bytes)
/// Layout:
///   [0..4]   magic "ALZ\x01"
///   [4]      version
///   [5]      mode
///   [6]      dtype
///   [7]      ndim
///   [8..24]  shape (4 x u32)
///   [24..32] original_size (u64)
#[derive(Debug)]
struct AlzHeader {
    magic: [u8; 4],          // "ALZ\x01"
    version: u8,             // Format version
    mode: CompressionMode,   // Compression mode
    dtype: u8,               // Data type (0=f32, 1=f64, 2=u8, 3=i16, etc.)
    ndim: u8,                // Number of dimensions
    shape: [u32; 4],         // Shape (up to 4D)
    original_size: u64,      // Original uncompressed size
}

impl AlzHeader {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        bytes.extend_from_slice(&self.magic);
        bytes.push(self.version);
        bytes.push(self.mode as u8);
        bytes.push(self.dtype);
        bytes.push(self.ndim);
        for s in &self.shape {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes.extend_from_slice(&self.original_size.to_le_bytes());
        debug_assert_eq!(bytes.len(), 32, "Header must be exactly 32 bytes");
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 32 {
            return None;
        }
        if &bytes[0..4] != &MAGIC {
            return None;
        }

        Some(AlzHeader {
            magic: [bytes[0], bytes[1], bytes[2], bytes[3]],
            version: bytes[4],
            mode: CompressionMode::from(bytes[5]),
            dtype: bytes[6],
            ndim: bytes[7],
            shape: [
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
                u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
                u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
                u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
            ],
            original_size: u64::from_le_bytes([
                bytes[24], bytes[25], bytes[26], bytes[27],
                bytes[28], bytes[29], bytes[30], bytes[31],
            ]),
        })
    }
}

// ============================================================================
// Commands Implementation
// ============================================================================

fn cmd_compress(
    input: PathBuf,
    output: PathBuf,
    quality: Quality,
    bits: u8,
    level: u32,
    raw: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("ALICE-Zip Compressor v0.1.0");
    println!("===========================");

    // Read input file
    let mut file = File::open(&input)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let original_size = data.len();
    println!("Input: {} ({} bytes)", input.display(), original_size);
    println!("Compression level: {}", level);

    let start = Instant::now();

    // Determine compression mode
    let (compressed_data, mode) = if raw {
        // Raw zlib mode
        let compressed = zlib_compress(&data, level)?;
        (compressed, CompressionMode::RawLzma)
    } else {
        // Try to interpret as f32 array
        let is_f32 = original_size % 4 == 0;

        if is_f32 && quality != Quality::Lossless {
            // Quantized compression
            let floats: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let (quantized, min_val, scale) = quantize_8bit(&floats);

            // Compress quantized data
            let compressed = zlib_compress(&quantized, level)?;

            // Build payload: [min_val(8)] [scale(8)] [compressed]
            let mut payload = Vec::with_capacity(16 + compressed.len());
            payload.extend_from_slice(&min_val.to_le_bytes());
            payload.extend_from_slice(&scale.to_le_bytes());
            payload.extend_from_slice(&compressed);

            let mode = if bits == 16 {
                CompressionMode::Quantized16
            } else {
                CompressionMode::Quantized8
            };

            (payload, mode)
        } else {
            // Lossless zlib
            let compressed = zlib_compress(&data, level)?;
            (compressed, CompressionMode::Lossless)
        }
    };

    let elapsed = start.elapsed();

    // Build header
    let header = AlzHeader {
        magic: MAGIC,
        version: FORMAT_VERSION,
        mode,
        dtype: 0, // f32
        ndim: 1,
        shape: [(original_size / 4) as u32, 0, 0, 0],
        original_size: original_size as u64,
    };

    // Write output
    let mut out_file = BufWriter::new(File::create(&output)?);
    out_file.write_all(&header.to_bytes())?;
    out_file.write_all(&compressed_data)?;
    out_file.flush()?;

    let output_size = 32 + compressed_data.len();
    let ratio = original_size as f64 / output_size as f64;

    println!("Output: {} ({} bytes)", output.display(), output_size);
    println!("Compression ratio: {:.2}x", ratio);
    println!("Mode: {:?}", mode);
    println!("Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}

fn cmd_decompress(input: PathBuf, output: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("ALICE-Zip Decompressor v0.1.0");
    println!("=============================");

    // Read input file
    let mut file = BufReader::new(File::open(&input)?);
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    println!("Input: {} ({} bytes)", input.display(), data.len());

    // Parse header
    let header = AlzHeader::from_bytes(&data)
        .ok_or("Invalid .alz file: bad header")?;

    let payload = &data[32..];

    let start = Instant::now();

    // Decompress based on mode
    let decompressed = match header.mode {
        CompressionMode::RawLzma | CompressionMode::Lossless => {
            zlib_decompress(payload)?
        }
        CompressionMode::Quantized8 | CompressionMode::Quantized16 => {
            // Validate payload length
            if payload.len() < 16 {
                return Err("Payload too short for quantized data".into());
            }

            // Parse quantization parameters
            let min_val = f64::from_le_bytes(payload[0..8].try_into()?);
            let scale = f64::from_le_bytes(payload[8..16].try_into()?);
            let compressed = &payload[16..];

            // Decompress
            let quantized = zlib_decompress(compressed)?;

            // Dequantize
            let floats = dequantize_8bit(&quantized, min_val, scale);

            // Convert to bytes
            floats.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect()
        }
        _ => {
            return Err("Unsupported compression mode".into());
        }
    };

    let elapsed = start.elapsed();

    // Write output
    let mut out_file = BufWriter::new(File::create(&output)?);
    out_file.write_all(&decompressed)?;
    out_file.flush()?;

    println!("Output: {} ({} bytes)", output.display(), decompressed.len());
    println!("Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}

fn cmd_info(file: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = File::open(&file)?;
    let mut header_bytes = [0u8; 32];

    let file_size = f.metadata()?.len();

    if f.read(&mut header_bytes)? < 32 {
        println!("File too small for .alz format");
        return Ok(());
    }

    if &header_bytes[0..4] == &MAGIC {
        // Valid .alz file
        let header = AlzHeader::from_bytes(&header_bytes)
            .ok_or("Invalid .alz file: failed to parse header")?;

        println!("ALICE-Zip File Information");
        println!("==========================");
        println!("File: {}", file.display());
        println!("Size: {} bytes", file_size);
        println!("Format version: {}", header.version);
        println!("Mode: {:?}", header.mode);
        println!("Original size: {} bytes", header.original_size);
        println!("Compression ratio: {:.2}x", header.original_size as f64 / file_size as f64);
        println!("Shape: {:?}", &header.shape[..header.ndim as usize]);
    } else {
        // Raw data file
        println!("Raw Data File Information");
        println!("=========================");
        println!("File: {}", file.display());
        println!("Size: {} bytes", file_size);

        if file_size % 4 == 0 {
            println!("Possible f32 array: {} elements", file_size / 4);
        }
        if file_size % 8 == 0 {
            println!("Possible f64 array: {} elements", file_size / 8);
        }
    }

    Ok(())
}

fn cmd_benchmark(input: PathBuf, iterations: u32, level: u32) -> Result<(), Box<dyn std::error::Error>> {
    println!("ALICE-Zip Benchmark v0.1.0");
    println!("==========================");

    let mut file = File::open(&input)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let original_size = data.len();
    println!("Input: {} ({} bytes)", input.display(), original_size);
    println!("Iterations: {}", iterations);
    println!("Compression level: {}", level);
    println!();

    // Benchmark zlib raw
    let mut zlib_times = Vec::new();
    let mut zlib_size = 0;
    for _ in 0..iterations {
        let start = Instant::now();
        let compressed = zlib_compress(&data, level)?;
        zlib_times.push(start.elapsed());
        zlib_size = compressed.len();
    }

    let avg_zlib = zlib_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / iterations as f64;
    println!("zlib (level {}):", level);
    println!("  Size: {} -> {} bytes ({:.2}x)", original_size, zlib_size, original_size as f64 / zlib_size as f64);
    println!("  Time: {:.2}ms avg", avg_zlib * 1000.0);
    println!("  Throughput: {:.2} MB/s", (original_size as f64 / 1024.0 / 1024.0) / avg_zlib);

    // Benchmark 8-bit quantization + zlib (if f32 data)
    if original_size % 4 == 0 {
        let floats: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut quant_times = Vec::new();
        let mut quant_size = 0;
        for _ in 0..iterations {
            let start = Instant::now();
            let (quantized, _, _) = quantize_8bit(&floats);
            let compressed = zlib_compress(&quantized, level)?;
            quant_times.push(start.elapsed());
            quant_size = 16 + compressed.len(); // header + compressed
        }

        let avg_quant = quant_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / iterations as f64;
        println!("\n8-bit Quantized + zlib (level {}):", level);
        println!("  Size: {} -> {} bytes ({:.2}x)", original_size, quant_size, original_size as f64 / quant_size as f64);
        println!("  Time: {:.2}ms avg", avg_quant * 1000.0);
        println!("  Throughput: {:.2} MB/s", (original_size as f64 / 1024.0 / 1024.0) / avg_quant);
    }

    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Compress { input, output, quality, bits, level, raw } => {
            cmd_compress(input, output, quality, bits, level, raw)
        }
        Commands::Decompress { input, output } => {
            cmd_decompress(input, output)
        }
        Commands::Info { file } => {
            cmd_info(file)
        }
        Commands::Benchmark { input, iterations, level } => {
            cmd_benchmark(input, iterations, level)
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
