/**
 * @file alice.h
 * @brief ALICE-Zip C API Header
 * @author Moroya Sakamoto
 * @version 2.2.0
 * @license MIT (Core) / Commercial (Pro features)
 *
 * This header provides C bindings for the libalice library.
 * Compatible with C, C++, C# (P/Invoke), and other FFI-capable languages.
 *
 * Game Industry Exception: Free with attribution for game development.
 * See LICENSE for details.
 */

#ifndef ALICE_H
#define ALICE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Platform-specific exports
 * ============================================================================ */

#if defined(_WIN32) || defined(_WIN64)
    #ifdef ALICE_EXPORTS
        #define ALICE_API __declspec(dllexport)
    #else
        #define ALICE_API __declspec(dllimport)
    #endif
#else
    #define ALICE_API __attribute__((visibility("default")))
#endif

/* ============================================================================
 * Error Codes
 * ============================================================================ */

/**
 * @brief Error codes returned by ALICE functions
 */
typedef enum AliceError {
    ALICE_SUCCESS = 0,           /**< Operation completed successfully */
    ALICE_ERROR_NULL_POINTER = 1,/**< A required pointer was null */
    ALICE_ERROR_INVALID_PARAM = 2,/**< Invalid parameter value */
    ALICE_ERROR_COMPRESSION = 3, /**< Compression operation failed */
    ALICE_ERROR_DECOMPRESSION = 4,/**< Decompression operation failed */
    ALICE_ERROR_ALLOCATION = 5,  /**< Memory allocation failed */
    ALICE_ERROR_INVALID_DATA = 6 /**< Input data is invalid or corrupted */
} AliceError;

/* ============================================================================
 * Buffer Types
 * ============================================================================ */

/**
 * @brief Buffer for returning byte arrays
 *
 * Must be freed with alice_free_buffer() when no longer needed.
 */
typedef struct AliceBuffer {
    uint8_t* data;    /**< Pointer to data */
    size_t len;       /**< Length of data in bytes */
    size_t capacity;  /**< Allocated capacity (internal use) */
} AliceBuffer;

/**
 * @brief Buffer for returning float arrays
 *
 * Must be freed with alice_free_float_buffer() when no longer needed.
 */
typedef struct AliceFloatBuffer {
    float* data;      /**< Pointer to float data */
    size_t len;       /**< Number of floats */
    size_t capacity;  /**< Allocated capacity (internal use) */
} AliceFloatBuffer;

/**
 * @brief Fourier coefficient for signal reconstruction
 */
typedef struct FourierCoefficient {
    size_t frequency; /**< Frequency bin index */
    float amplitude;  /**< Amplitude of this component */
    float phase;      /**< Phase in radians */
} FourierCoefficient;

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * @brief Free a buffer allocated by libalice
 * @param buffer Pointer to buffer to free
 */
ALICE_API void alice_free_buffer(AliceBuffer* buffer);

/**
 * @brief Free a float buffer allocated by libalice
 * @param buffer Pointer to buffer to free
 */
ALICE_API void alice_free_float_buffer(AliceFloatBuffer* buffer);

/* ============================================================================
 * Version Information
 * ============================================================================ */

/**
 * @brief Get the library version string
 * @return Null-terminated version string (e.g., "2.2.0")
 */
ALICE_API const char* alice_version(void);

/**
 * @brief Get version numbers as integers
 * @param major Output for major version (can be NULL)
 * @param minor Output for minor version (can be NULL)
 * @param patch Output for patch version (can be NULL)
 */
ALICE_API void alice_version_numbers(uint32_t* major, uint32_t* minor, uint32_t* patch);

/**
 * @brief Get the last error message
 * @return Pointer to error message or NULL if no error
 * @note The string is valid until the next FFI call on the same thread
 */
ALICE_API const char* alice_get_last_error(void);

/* ============================================================================
 * Perlin Noise Generation
 * ============================================================================ */

/**
 * @brief Generate 2D Perlin noise texture
 *
 * @param width Width of the output texture
 * @param height Height of the output texture
 * @param seed Random seed for reproducibility
 * @param scale Noise scale (larger = more zoomed out)
 * @param octaves Number of octaves for fractal noise (1-8 recommended)
 * @param out_buffer Output buffer (will be allocated, caller must free)
 * @return ALICE_SUCCESS on success, error code otherwise
 *
 * @code
 * AliceFloatBuffer buffer;
 * if (alice_perlin_2d(256, 256, 42, 10.0f, 4, &buffer) == ALICE_SUCCESS) {
 *     // Use buffer.data (256*256 floats in row-major order)
 *     alice_free_float_buffer(&buffer);
 * }
 * @endcode
 */
ALICE_API AliceError alice_perlin_2d(
    size_t width,
    size_t height,
    uint64_t seed,
    float scale,
    uint32_t octaves,
    AliceFloatBuffer* out_buffer
);

/**
 * @brief Generate advanced 2D Perlin noise with persistence and lacunarity
 *
 * @param width Width of the output texture
 * @param height Height of the output texture
 * @param seed Random seed for reproducibility
 * @param scale Noise scale
 * @param octaves Number of octaves
 * @param persistence Amplitude multiplier per octave (0.5 typical)
 * @param lacunarity Frequency multiplier per octave (2.0 typical)
 * @param out_buffer Output buffer
 * @return ALICE_SUCCESS on success
 */
ALICE_API AliceError alice_perlin_advanced(
    size_t width,
    size_t height,
    uint64_t seed,
    float scale,
    uint32_t octaves,
    float persistence,
    float lacunarity,
    AliceFloatBuffer* out_buffer
);

/* ============================================================================
 * Fourier / Sine Wave Generation
 * ============================================================================ */

/**
 * @brief Generate a sine wave
 *
 * @param n Number of samples
 * @param frequency Frequency in cycles per signal length
 * @param amplitude Wave amplitude
 * @param phase Phase offset in radians
 * @param dc_offset DC offset to add
 * @param out_buffer Output buffer
 * @return ALICE_SUCCESS on success
 */
ALICE_API AliceError alice_sine_wave(
    size_t n,
    float frequency,
    float amplitude,
    float phase,
    float dc_offset,
    AliceFloatBuffer* out_buffer
);

/**
 * @brief Generate signal from Fourier coefficients
 *
 * @param n Number of samples to generate
 * @param coefficients Array of Fourier coefficients
 * @param num_coefficients Number of coefficients
 * @param dc_offset DC offset to add
 * @param out_buffer Output buffer
 * @return ALICE_SUCCESS on success
 */
ALICE_API AliceError alice_fourier_generate(
    size_t n,
    const FourierCoefficient* coefficients,
    size_t num_coefficients,
    float dc_offset,
    AliceFloatBuffer* out_buffer
);

/* ============================================================================
 * Polynomial Generation
 * ============================================================================ */

/**
 * @brief Generate polynomial data: y = c0 + c1*x + c2*x^2 + ...
 *
 * @param n Number of samples (x ranges from 0 to n-1)
 * @param coefficients Polynomial coefficients [c0, c1, c2, ...]
 * @param num_coefficients Number of coefficients
 * @param out_buffer Output buffer
 * @return ALICE_SUCCESS on success
 */
ALICE_API AliceError alice_polynomial_generate(
    size_t n,
    const double* coefficients,
    size_t num_coefficients,
    AliceFloatBuffer* out_buffer
);

/* ============================================================================
 * Compression Functions
 * ============================================================================ */

/**
 * @brief Compress data using LZMA algorithm
 *
 * @param data Input data to compress
 * @param len Length of input data
 * @param preset Compression preset (0-9, higher = better compression, slower)
 * @param out_buffer Output buffer for compressed data
 * @return ALICE_SUCCESS on success
 */
ALICE_API AliceError alice_lzma_compress(
    const uint8_t* data,
    size_t len,
    uint32_t preset,
    AliceBuffer* out_buffer
);

/**
 * @brief Decompress LZMA data
 *
 * @param data Compressed data
 * @param len Length of compressed data
 * @param out_buffer Output buffer for decompressed data
 * @return ALICE_SUCCESS on success
 */
ALICE_API AliceError alice_lzma_decompress(
    const uint8_t* data,
    size_t len,
    AliceBuffer* out_buffer
);

/**
 * @brief Compress data using zlib algorithm
 *
 * @param data Input data to compress
 * @param len Length of input data
 * @param level Compression level (1-9)
 * @param out_buffer Output buffer for compressed data
 * @return ALICE_SUCCESS on success
 */
ALICE_API AliceError alice_zlib_compress(
    const uint8_t* data,
    size_t len,
    uint32_t level,
    AliceBuffer* out_buffer
);

/**
 * @brief Decompress zlib data
 *
 * @param data Compressed data
 * @param len Length of compressed data
 * @param out_buffer Output buffer for decompressed data
 * @return ALICE_SUCCESS on success
 */
ALICE_API AliceError alice_zlib_decompress(
    const uint8_t* data,
    size_t len,
    AliceBuffer* out_buffer
);

/**
 * @brief Compress float residuals with quantization
 *
 * Quantizes floats to specified bit depth and compresses with LZMA.
 * Useful for lossy compression of texture/signal residuals.
 *
 * @param residual Float residual data
 * @param len Number of floats
 * @param bits Quantization bits (8, 12, or 16)
 * @param lzma_preset LZMA compression preset
 * @param out_buffer Output buffer for compressed data
 * @return ALICE_SUCCESS on success
 */
ALICE_API AliceError alice_residual_compress(
    const float* residual,
    size_t len,
    uint8_t bits,
    uint32_t lzma_preset,
    AliceBuffer* out_buffer
);

/**
 * @brief Decompress quantized residuals
 *
 * @param data Compressed residual data
 * @param len Length of compressed data
 * @param out_buffer Output buffer for decompressed floats
 * @return ALICE_SUCCESS on success
 */
ALICE_API AliceError alice_residual_decompress(
    const uint8_t* data,
    size_t len,
    AliceFloatBuffer* out_buffer
);

#ifdef __cplusplus
}
#endif

#endif /* ALICE_H */
