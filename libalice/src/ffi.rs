//! C FFI bindings for libalice
//!
//! This module provides C-compatible function exports for use with
//! C, C++, C# (P/Invoke), and other languages via FFI.
//!
//! # Safety
//! All functions in this module use raw pointers and are inherently unsafe.
//! Callers must ensure proper memory management.

use std::ffi::c_char;
use std::ptr;
use std::slice;

use crate::{generators, compression};

// ============================================================================
// Error Handling
// ============================================================================

/// Error codes returned by FFI functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliceError {
    Success = 0,
    NullPointer = 1,
    InvalidParameter = 2,
    CompressionError = 3,
    DecompressionError = 4,
    AllocationError = 5,
    InvalidData = 6,
}

// Thread-local error message storage
thread_local! {
    static LAST_ERROR: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = msg.to_string();
    });
}

/// Get the last error message.
/// Returns a pointer to a null-terminated string. The string is valid until
/// the next FFI call on the same thread.
#[no_mangle]
pub extern "C" fn alice_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        let msg = e.borrow();
        if msg.is_empty() {
            ptr::null()
        } else {
            msg.as_ptr() as *const c_char
        }
    })
}

// ============================================================================
// Memory Management
// ============================================================================

/// Buffer structure for returning variable-length data
#[repr(C)]
pub struct AliceBuffer {
    pub data: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

impl AliceBuffer {
    fn new(data: Vec<u8>) -> Self {
        let mut data = data.into_boxed_slice();
        let ptr = data.as_mut_ptr();
        let len = data.len();
        std::mem::forget(data);
        AliceBuffer {
            data: ptr,
            len,
            capacity: len,
        }
    }

    fn null() -> Self {
        AliceBuffer {
            data: ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }
}

/// Float buffer for returning float arrays
#[repr(C)]
pub struct AliceFloatBuffer {
    pub data: *mut f32,
    pub len: usize,
    pub capacity: usize,
}

impl AliceFloatBuffer {
    fn new(data: Vec<f32>) -> Self {
        let mut data = data.into_boxed_slice();
        let ptr = data.as_mut_ptr();
        let len = data.len();
        std::mem::forget(data);
        AliceFloatBuffer {
            data: ptr,
            len,
            capacity: len,
        }
    }

    fn null() -> Self {
        AliceFloatBuffer {
            data: ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }
}

/// Free a buffer allocated by libalice
#[no_mangle]
pub extern "C" fn alice_free_buffer(buffer: *mut AliceBuffer) {
    if buffer.is_null() {
        return;
    }
    unsafe {
        let buf = &*buffer;
        if !buf.data.is_null() && buf.capacity > 0 {
            let _ = Vec::from_raw_parts(buf.data, buf.len, buf.capacity);
        }
    }
}

/// Free a float buffer allocated by libalice
#[no_mangle]
pub extern "C" fn alice_free_float_buffer(buffer: *mut AliceFloatBuffer) {
    if buffer.is_null() {
        return;
    }
    unsafe {
        let buf = &*buffer;
        if !buf.data.is_null() && buf.capacity > 0 {
            let _ = Vec::from_raw_parts(buf.data, buf.len, buf.capacity);
        }
    }
}

// ============================================================================
// Version Info
// ============================================================================

/// Get the library version string
#[no_mangle]
pub extern "C" fn alice_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

/// Get the library version as integers
#[no_mangle]
pub extern "C" fn alice_version_numbers(major: *mut u32, minor: *mut u32, patch: *mut u32) {
    if !major.is_null() {
        unsafe { *major = env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0); }
    }
    if !minor.is_null() {
        unsafe { *minor = env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0); }
    }
    if !patch.is_null() {
        unsafe { *patch = env!("CARGO_PKG_VERSION_PATCH").parse().unwrap_or(0); }
    }
}

// ============================================================================
// Perlin Noise Generation
// ============================================================================

/// Generate 2D Perlin noise
///
/// # Parameters
/// - `width`: Width of the output texture
/// - `height`: Height of the output texture
/// - `seed`: Random seed for reproducibility
/// - `scale`: Noise scale (larger = more zoomed out)
/// - `octaves`: Number of octaves for fractal noise
/// - `out_buffer`: Output buffer (will be allocated by the function)
///
/// # Returns
/// AliceError::Success on success, error code otherwise
#[no_mangle]
pub extern "C" fn alice_perlin_2d(
    width: usize,
    height: usize,
    seed: u64,
    scale: f32,
    octaves: u32,
    out_buffer: *mut AliceFloatBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if width == 0 || height == 0 {
        set_last_error("Width and height must be greater than 0");
        return AliceError::InvalidParameter;
    }
    if scale <= 0.0 {
        set_last_error("Scale must be greater than 0");
        return AliceError::InvalidParameter;
    }

    let data = generators::generate_perlin_2d(width, height, seed, scale, octaves);
    unsafe {
        *out_buffer = AliceFloatBuffer::new(data);
    }
    AliceError::Success
}

/// Generate advanced 2D Perlin noise with persistence and lacunarity
#[no_mangle]
pub extern "C" fn alice_perlin_advanced(
    width: usize,
    height: usize,
    seed: u64,
    scale: f32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    out_buffer: *mut AliceFloatBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if width == 0 || height == 0 {
        set_last_error("Width and height must be greater than 0");
        return AliceError::InvalidParameter;
    }

    let data = generators::generate_perlin_advanced(
        width, height, seed, scale, octaves, persistence, lacunarity
    );
    unsafe {
        *out_buffer = AliceFloatBuffer::new(data);
    }
    AliceError::Success
}

// ============================================================================
// Fourier / Sine Wave Generation
// ============================================================================

/// Generate a sine wave
#[no_mangle]
pub extern "C" fn alice_sine_wave(
    n: usize,
    frequency: f32,
    amplitude: f32,
    phase: f32,
    dc_offset: f32,
    out_buffer: *mut AliceFloatBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if n == 0 {
        set_last_error("n must be greater than 0");
        return AliceError::InvalidParameter;
    }

    let data = generators::generate_sine_wave(n, frequency, amplitude, phase, dc_offset);
    unsafe {
        *out_buffer = AliceFloatBuffer::new(data);
    }
    AliceError::Success
}

/// Fourier coefficient for signal reconstruction
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FourierCoefficient {
    pub frequency: usize,
    pub amplitude: f32,
    pub phase: f32,
}

/// Generate signal from Fourier coefficients
#[no_mangle]
pub extern "C" fn alice_fourier_generate(
    n: usize,
    coefficients: *const FourierCoefficient,
    num_coefficients: usize,
    dc_offset: f32,
    out_buffer: *mut AliceFloatBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if coefficients.is_null() && num_coefficients > 0 {
        set_last_error("Coefficients pointer is null");
        return AliceError::NullPointer;
    }

    let coeffs: Vec<(usize, f32, f32)> = if num_coefficients > 0 {
        unsafe {
            slice::from_raw_parts(coefficients, num_coefficients)
                .iter()
                .map(|c| (c.frequency, c.amplitude, c.phase))
                .collect()
        }
    } else {
        Vec::new()
    };

    let data = generators::generate_from_coefficients(n, &coeffs, dc_offset);
    unsafe {
        *out_buffer = AliceFloatBuffer::new(data);
    }
    AliceError::Success
}

// ============================================================================
// Polynomial Generation
// ============================================================================

/// Generate polynomial data: y = c0 + c1*x + c2*x^2 + ...
#[no_mangle]
pub extern "C" fn alice_polynomial_generate(
    n: usize,
    coefficients: *const f64,
    num_coefficients: usize,
    out_buffer: *mut AliceFloatBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if coefficients.is_null() && num_coefficients > 0 {
        set_last_error("Coefficients pointer is null");
        return AliceError::NullPointer;
    }

    let coeffs: Vec<f64> = if num_coefficients > 0 {
        unsafe { slice::from_raw_parts(coefficients, num_coefficients).to_vec() }
    } else {
        Vec::new()
    };

    let data = generators::generate_polynomial(n, &coeffs);
    unsafe {
        *out_buffer = AliceFloatBuffer::new(data);
    }
    AliceError::Success
}

// ============================================================================
// Compression Functions
// ============================================================================

/// Compress data using LZMA
#[no_mangle]
pub extern "C" fn alice_lzma_compress(
    data: *const u8,
    len: usize,
    preset: u32,
    out_buffer: *mut AliceBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if data.is_null() && len > 0 {
        set_last_error("Data pointer is null");
        return AliceError::NullPointer;
    }

    let input = if len > 0 {
        unsafe { slice::from_raw_parts(data, len) }
    } else {
        &[]
    };

    match compression::lzma_compress(input, preset) {
        Ok(compressed) => {
            unsafe { *out_buffer = AliceBuffer::new(compressed); }
            AliceError::Success
        }
        Err(e) => {
            set_last_error(&format!("LZMA compression failed: {}", e));
            unsafe { *out_buffer = AliceBuffer::null(); }
            AliceError::CompressionError
        }
    }
}

/// Decompress LZMA data
#[no_mangle]
pub extern "C" fn alice_lzma_decompress(
    data: *const u8,
    len: usize,
    out_buffer: *mut AliceBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if data.is_null() && len > 0 {
        set_last_error("Data pointer is null");
        return AliceError::NullPointer;
    }

    let input = if len > 0 {
        unsafe { slice::from_raw_parts(data, len) }
    } else {
        &[]
    };

    match compression::lzma_decompress(input) {
        Ok(decompressed) => {
            unsafe { *out_buffer = AliceBuffer::new(decompressed); }
            AliceError::Success
        }
        Err(e) => {
            set_last_error(&format!("LZMA decompression failed: {}", e));
            unsafe { *out_buffer = AliceBuffer::null(); }
            AliceError::DecompressionError
        }
    }
}

/// Compress data using zlib
#[no_mangle]
pub extern "C" fn alice_zlib_compress(
    data: *const u8,
    len: usize,
    level: u32,
    out_buffer: *mut AliceBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if data.is_null() && len > 0 {
        set_last_error("Data pointer is null");
        return AliceError::NullPointer;
    }

    let input = if len > 0 {
        unsafe { slice::from_raw_parts(data, len) }
    } else {
        &[]
    };

    match compression::zlib_compress(input, level) {
        Ok(compressed) => {
            unsafe { *out_buffer = AliceBuffer::new(compressed); }
            AliceError::Success
        }
        Err(e) => {
            set_last_error(&format!("zlib compression failed: {}", e));
            unsafe { *out_buffer = AliceBuffer::null(); }
            AliceError::CompressionError
        }
    }
}

/// Decompress zlib data
#[no_mangle]
pub extern "C" fn alice_zlib_decompress(
    data: *const u8,
    len: usize,
    out_buffer: *mut AliceBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if data.is_null() && len > 0 {
        set_last_error("Data pointer is null");
        return AliceError::NullPointer;
    }

    let input = if len > 0 {
        unsafe { slice::from_raw_parts(data, len) }
    } else {
        &[]
    };

    match compression::zlib_decompress(input) {
        Ok(decompressed) => {
            unsafe { *out_buffer = AliceBuffer::new(decompressed); }
            AliceError::Success
        }
        Err(e) => {
            set_last_error(&format!("zlib decompression failed: {}", e));
            unsafe { *out_buffer = AliceBuffer::null(); }
            AliceError::DecompressionError
        }
    }
}

/// Compress float residuals with quantization
#[no_mangle]
pub extern "C" fn alice_residual_compress(
    residual: *const f32,
    len: usize,
    bits: u8,
    lzma_preset: u32,
    out_buffer: *mut AliceBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if residual.is_null() && len > 0 {
        set_last_error("Residual pointer is null");
        return AliceError::NullPointer;
    }

    let input: Vec<f32> = if len > 0 {
        unsafe { slice::from_raw_parts(residual, len).to_vec() }
    } else {
        Vec::new()
    };

    match compression::compress_residual_quantized(&input, bits, lzma_preset) {
        Ok(compressed) => {
            unsafe { *out_buffer = AliceBuffer::new(compressed); }
            AliceError::Success
        }
        Err(e) => {
            set_last_error(&format!("Residual compression failed: {}", e));
            unsafe { *out_buffer = AliceBuffer::null(); }
            AliceError::CompressionError
        }
    }
}

/// Decompress quantized residuals
#[no_mangle]
pub extern "C" fn alice_residual_decompress(
    data: *const u8,
    len: usize,
    out_buffer: *mut AliceFloatBuffer,
) -> AliceError {
    if out_buffer.is_null() {
        set_last_error("Output buffer pointer is null");
        return AliceError::NullPointer;
    }
    if data.is_null() && len > 0 {
        set_last_error("Data pointer is null");
        return AliceError::NullPointer;
    }

    let input = if len > 0 {
        unsafe { slice::from_raw_parts(data, len) }
    } else {
        &[]
    };

    match compression::decompress_residual_quantized(input) {
        Ok(decompressed) => {
            unsafe { *out_buffer = AliceFloatBuffer::new(decompressed); }
            AliceError::Success
        }
        Err(e) => {
            set_last_error(&format!("Residual decompression failed: {}", e));
            unsafe { *out_buffer = AliceFloatBuffer::null(); }
            AliceError::DecompressionError
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = alice_version();
        assert!(!version.is_null());
    }

    #[test]
    fn test_perlin_2d() {
        let mut buffer = AliceFloatBuffer::null();
        let result = alice_perlin_2d(64, 64, 42, 10.0, 4, &mut buffer);
        assert_eq!(result, AliceError::Success);
        assert!(!buffer.data.is_null());
        assert_eq!(buffer.len, 64 * 64);
        alice_free_float_buffer(&mut buffer);
    }

    #[test]
    fn test_lzma_roundtrip() {
        let data = b"Hello, World! This is a test for LZMA compression.";
        let mut compressed = AliceBuffer::null();
        let result = alice_lzma_compress(data.as_ptr(), data.len(), 6, &mut compressed);
        assert_eq!(result, AliceError::Success);

        let mut decompressed = AliceBuffer::null();
        let result = alice_lzma_decompress(compressed.data, compressed.len, &mut decompressed);
        assert_eq!(result, AliceError::Success);

        let output = unsafe { slice::from_raw_parts(decompressed.data, decompressed.len) };
        assert_eq!(output, data);

        alice_free_buffer(&mut compressed);
        alice_free_buffer(&mut decompressed);
    }

    #[test]
    fn test_null_pointer_handling() {
        let result = alice_perlin_2d(64, 64, 42, 10.0, 4, ptr::null_mut());
        assert_eq!(result, AliceError::NullPointer);
    }
}
