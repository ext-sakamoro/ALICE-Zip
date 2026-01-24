/**
 * ALICE-Zip C++ Tests
 */

#include "alice.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    test_##name(); \
    std::cout << "PASSED" << std::endl; \
} while(0)

TEST(version) {
    auto version = alice::version();
    assert(!version.empty());

    auto [major, minor, patch] = alice::version_numbers();
    assert(major >= 2);
}

TEST(perlin_2d) {
    auto noise = alice::perlin_2d(64, 64, 42, 10.0f, 4);
    assert(noise.size() == 64 * 64);

    // Check values are in expected range
    for (float v : noise) {
        assert(v >= 0.0f && v <= 1.0f);
    }
}

TEST(perlin_advanced) {
    auto noise = alice::perlin_advanced(64, 64, 42, 10.0f, 4, 0.5f, 2.0f);
    assert(noise.size() == 64 * 64);
}

TEST(perlin_deterministic) {
    auto noise1 = alice::perlin_2d(32, 32, 123, 10.0f, 4);
    auto noise2 = alice::perlin_2d(32, 32, 123, 10.0f, 4);
    assert(noise1 == noise2);
}

TEST(sine_wave) {
    auto wave = alice::sine_wave(100, 5.0f, 1.0f, 0.0f, 0.0f);
    assert(wave.size() == 100);
}

TEST(polynomial) {
    std::vector<double> coeffs = {1.0, 2.0, 3.0};  // 1 + 2x + 3x^2
    auto poly = alice::polynomial_generate(100, coeffs);
    assert(poly.size() == 100);
}

TEST(lzma_roundtrip) {
    std::vector<uint8_t> data = {'H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!'};

    auto compressed = alice::lzma_compress(data, 6);
    auto decompressed = alice::lzma_decompress(compressed);

    assert(data == decompressed);
}

TEST(zlib_roundtrip) {
    std::vector<uint8_t> data = {'H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!'};

    auto compressed = alice::zlib_compress(data, 6);
    auto decompressed = alice::zlib_decompress(compressed);

    assert(data == decompressed);
}

TEST(residual_roundtrip) {
    std::vector<float> residual(100);
    for (size_t i = 0; i < residual.size(); i++) {
        residual[i] = static_cast<float>(i) / residual.size();
    }

    auto compressed = alice::residual_compress(residual, 8, 6);
    auto decompressed = alice::residual_decompress(compressed);

    assert(residual.size() == decompressed.size());

    // Check error is within quantization bounds
    for (size_t i = 0; i < residual.size(); i++) {
        float error = std::abs(residual[i] - decompressed[i]);
        assert(error < 0.01f);  // 8-bit quantization error bound
    }
}

TEST(error_handling) {
    bool caught = false;
    try {
        // This should fail with invalid parameters
        alice::perlin_2d(0, 0, 42, 10.0f, 4);
    } catch (const alice::Exception& e) {
        caught = true;
        assert(e.error_code != ALICE_SUCCESS);
    }
    assert(caught);
}

int main() {
    std::cout << "ALICE-Zip C++ Tests" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << std::endl;

    RUN_TEST(version);
    RUN_TEST(perlin_2d);
    RUN_TEST(perlin_advanced);
    RUN_TEST(perlin_deterministic);
    RUN_TEST(sine_wave);
    RUN_TEST(polynomial);
    RUN_TEST(lzma_roundtrip);
    RUN_TEST(zlib_roundtrip);
    RUN_TEST(residual_roundtrip);
    RUN_TEST(error_handling);

    std::cout << std::endl;
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
