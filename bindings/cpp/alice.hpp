/**
 * @file alice.hpp
 * @brief ALICE-Zip C++ Header-Only Wrapper
 * @author Moroya Sakamoto
 * @version 2.2.0
 * @license MIT (Core) / Game Industry Exception available
 *
 * Modern C++ wrapper for libalice providing RAII, exceptions, and STL containers.
 *
 * Usage:
 *   #include "alice.hpp"
 *
 *   // Generate Perlin noise
 *   auto noise = alice::perlin_2d(256, 256, 42, 10.0f, 4);
 *
 *   // Compress data
 *   auto compressed = alice::lzma_compress(data);
 */

#ifndef ALICE_HPP
#define ALICE_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

// Include the C header
extern "C" {
#include "alice.h"
}

namespace alice {

// ============================================================================
// Exception Classes
// ============================================================================

/**
 * @brief Exception thrown when an ALICE operation fails
 */
class Exception : public std::runtime_error {
public:
    AliceError error_code;

    explicit Exception(AliceError code, const std::string& message = "")
        : std::runtime_error(message.empty() ? get_default_message(code) : message),
          error_code(code) {}

private:
    static std::string get_default_message(AliceError code) {
        switch (code) {
            case ALICE_ERROR_NULL_POINTER: return "A required pointer was null";
            case ALICE_ERROR_INVALID_PARAM: return "Invalid parameter value";
            case ALICE_ERROR_COMPRESSION: return "Compression operation failed";
            case ALICE_ERROR_DECOMPRESSION: return "Decompression operation failed";
            case ALICE_ERROR_ALLOCATION: return "Memory allocation failed";
            case ALICE_ERROR_INVALID_DATA: return "Input data is invalid or corrupted";
            default: return "Unknown error";
        }
    }
};

// ============================================================================
// RAII Buffer Wrappers
// ============================================================================

namespace detail {

/**
 * @brief RAII wrapper for AliceBuffer
 */
class BufferGuard {
public:
    AliceBuffer buffer{nullptr, 0, 0};

    ~BufferGuard() {
        if (buffer.data) {
            alice_free_buffer(&buffer);
        }
    }

    std::vector<uint8_t> to_vector() {
        if (!buffer.data || buffer.len == 0) {
            return {};
        }
        return std::vector<uint8_t>(buffer.data, buffer.data + buffer.len);
    }

    // Non-copyable
    BufferGuard(const BufferGuard&) = delete;
    BufferGuard& operator=(const BufferGuard&) = delete;

    BufferGuard() = default;
    BufferGuard(BufferGuard&& other) noexcept : buffer(other.buffer) {
        other.buffer = {nullptr, 0, 0};
    }
};

/**
 * @brief RAII wrapper for AliceFloatBuffer
 */
class FloatBufferGuard {
public:
    AliceFloatBuffer buffer{nullptr, 0, 0};

    ~FloatBufferGuard() {
        if (buffer.data) {
            alice_free_float_buffer(&buffer);
        }
    }

    std::vector<float> to_vector() {
        if (!buffer.data || buffer.len == 0) {
            return {};
        }
        return std::vector<float>(buffer.data, buffer.data + buffer.len);
    }

    // Non-copyable
    FloatBufferGuard(const FloatBufferGuard&) = delete;
    FloatBufferGuard& operator=(const FloatBufferGuard&) = delete;

    FloatBufferGuard() = default;
    FloatBufferGuard(FloatBufferGuard&& other) noexcept : buffer(other.buffer) {
        other.buffer = {nullptr, 0, 0};
    }
};

inline void check_error(AliceError error) {
    if (error != ALICE_SUCCESS) {
        const char* msg = alice_get_last_error();
        throw Exception(error, msg ? msg : "");
    }
}

} // namespace detail

// ============================================================================
// Version Information
// ============================================================================

/**
 * @brief Get the library version string
 */
inline std::string version() {
    return alice_version();
}

/**
 * @brief Get version numbers
 */
inline std::tuple<uint32_t, uint32_t, uint32_t> version_numbers() {
    uint32_t major, minor, patch;
    alice_version_numbers(&major, &minor, &patch);
    return {major, minor, patch};
}

// ============================================================================
// Perlin Noise Generation
// ============================================================================

/**
 * @brief Generate 2D Perlin noise
 *
 * @param width Width of the output texture
 * @param height Height of the output texture
 * @param seed Random seed for reproducibility
 * @param scale Noise scale (larger = more zoomed out)
 * @param octaves Number of octaves for fractal noise
 * @return Vector of float values (row-major order)
 */
inline std::vector<float> perlin_2d(
    size_t width, size_t height,
    uint64_t seed = 42,
    float scale = 10.0f,
    uint32_t octaves = 4)
{
    detail::FloatBufferGuard guard;
    detail::check_error(alice_perlin_2d(width, height, seed, scale, octaves, &guard.buffer));
    return guard.to_vector();
}

/**
 * @brief Generate advanced 2D Perlin noise with persistence and lacunarity
 */
inline std::vector<float> perlin_advanced(
    size_t width, size_t height,
    uint64_t seed = 42,
    float scale = 10.0f,
    uint32_t octaves = 4,
    float persistence = 0.5f,
    float lacunarity = 2.0f)
{
    detail::FloatBufferGuard guard;
    detail::check_error(alice_perlin_advanced(
        width, height, seed, scale, octaves, persistence, lacunarity, &guard.buffer));
    return guard.to_vector();
}

// ============================================================================
// Signal Generation
// ============================================================================

/**
 * @brief Generate a sine wave
 */
inline std::vector<float> sine_wave(
    size_t n,
    float frequency,
    float amplitude = 1.0f,
    float phase = 0.0f,
    float dc_offset = 0.0f)
{
    detail::FloatBufferGuard guard;
    detail::check_error(alice_sine_wave(n, frequency, amplitude, phase, dc_offset, &guard.buffer));
    return guard.to_vector();
}

/**
 * @brief Generate signal from Fourier coefficients
 */
inline std::vector<float> fourier_generate(
    size_t n,
    const std::vector<FourierCoefficient>& coefficients,
    float dc_offset = 0.0f)
{
    detail::FloatBufferGuard guard;
    detail::check_error(alice_fourier_generate(
        n, coefficients.data(), coefficients.size(), dc_offset, &guard.buffer));
    return guard.to_vector();
}

/**
 * @brief Generate polynomial data: y = c0 + c1*x + c2*x^2 + ...
 */
inline std::vector<float> polynomial_generate(
    size_t n,
    const std::vector<double>& coefficients)
{
    detail::FloatBufferGuard guard;
    detail::check_error(alice_polynomial_generate(
        n, coefficients.data(), coefficients.size(), &guard.buffer));
    return guard.to_vector();
}

// ============================================================================
// Compression Functions
// ============================================================================

/**
 * @brief Compress data using LZMA
 */
inline std::vector<uint8_t> lzma_compress(
    const std::vector<uint8_t>& data,
    uint32_t preset = 6)
{
    detail::BufferGuard guard;
    detail::check_error(alice_lzma_compress(data.data(), data.size(), preset, &guard.buffer));
    return guard.to_vector();
}

/**
 * @brief Compress data using LZMA (raw pointer version)
 */
inline std::vector<uint8_t> lzma_compress(
    const uint8_t* data, size_t len,
    uint32_t preset = 6)
{
    detail::BufferGuard guard;
    detail::check_error(alice_lzma_compress(data, len, preset, &guard.buffer));
    return guard.to_vector();
}

/**
 * @brief Decompress LZMA data
 */
inline std::vector<uint8_t> lzma_decompress(const std::vector<uint8_t>& data)
{
    detail::BufferGuard guard;
    detail::check_error(alice_lzma_decompress(data.data(), data.size(), &guard.buffer));
    return guard.to_vector();
}

/**
 * @brief Compress data using zlib
 */
inline std::vector<uint8_t> zlib_compress(
    const std::vector<uint8_t>& data,
    uint32_t level = 6)
{
    detail::BufferGuard guard;
    detail::check_error(alice_zlib_compress(data.data(), data.size(), level, &guard.buffer));
    return guard.to_vector();
}

/**
 * @brief Decompress zlib data
 */
inline std::vector<uint8_t> zlib_decompress(const std::vector<uint8_t>& data)
{
    detail::BufferGuard guard;
    detail::check_error(alice_zlib_decompress(data.data(), data.size(), &guard.buffer));
    return guard.to_vector();
}

/**
 * @brief Compress float residuals with quantization
 */
inline std::vector<uint8_t> residual_compress(
    const std::vector<float>& residual,
    uint8_t bits = 8,
    uint32_t lzma_preset = 6)
{
    detail::BufferGuard guard;
    detail::check_error(alice_residual_compress(
        residual.data(), residual.size(), bits, lzma_preset, &guard.buffer));
    return guard.to_vector();
}

/**
 * @brief Decompress quantized residuals
 */
inline std::vector<float> residual_decompress(const std::vector<uint8_t>& data)
{
    detail::FloatBufferGuard guard;
    detail::check_error(alice_residual_decompress(data.data(), data.size(), &guard.buffer));
    return guard.to_vector();
}

} // namespace alice

#endif // ALICE_HPP
