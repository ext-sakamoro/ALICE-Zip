/**
 * ALICE-Zip C++ Example: Compression
 *
 * This example demonstrates how to compress and decompress data
 * using the ALICE-Zip C++ bindings.
 */

#include "alice.hpp"
#include <iostream>
#include <string>
#include <cstring>

int main() {
    std::cout << "ALICE-Zip C++ Compression Example" << std::endl;
    std::cout << "Version: " << alice::version() << std::endl;
    std::cout << std::endl;

    try {
        // Create test data
        std::string text = "Hello, World! This is a test of ALICE-Zip compression. "
                          "ALICE-Zip provides high-performance procedural compression "
                          "for game development and other applications.";

        // Repeat text to make it more compressible
        std::string repeated;
        for (int i = 0; i < 100; i++) {
            repeated += text;
        }

        std::vector<uint8_t> data(repeated.begin(), repeated.end());

        std::cout << "Original size: " << data.size() << " bytes" << std::endl;

        // LZMA compression
        std::cout << std::endl;
        std::cout << "=== LZMA Compression ===" << std::endl;

        auto lzma_compressed = alice::lzma_compress(data, 6);
        std::cout << "Compressed size: " << lzma_compressed.size() << " bytes" << std::endl;
        std::cout << "Ratio: " << (100.0 * lzma_compressed.size() / data.size()) << "%" << std::endl;

        auto lzma_decompressed = alice::lzma_decompress(lzma_compressed);
        std::cout << "Decompressed size: " << lzma_decompressed.size() << " bytes" << std::endl;

        // Verify
        if (data == lzma_decompressed) {
            std::cout << "Verification: PASSED" << std::endl;
        } else {
            std::cout << "Verification: FAILED" << std::endl;
        }

        // zlib compression
        std::cout << std::endl;
        std::cout << "=== zlib Compression ===" << std::endl;

        auto zlib_compressed = alice::zlib_compress(data, 6);
        std::cout << "Compressed size: " << zlib_compressed.size() << " bytes" << std::endl;
        std::cout << "Ratio: " << (100.0 * zlib_compressed.size() / data.size()) << "%" << std::endl;

        auto zlib_decompressed = alice::zlib_decompress(zlib_compressed);
        std::cout << "Decompressed size: " << zlib_decompressed.size() << " bytes" << std::endl;

        // Verify
        if (data == zlib_decompressed) {
            std::cout << "Verification: PASSED" << std::endl;
        } else {
            std::cout << "Verification: FAILED" << std::endl;
        }

        // Residual compression (lossy)
        std::cout << std::endl;
        std::cout << "=== Residual Compression (Lossy) ===" << std::endl;

        // Generate some float data (simulating texture residuals)
        std::vector<float> residual(1024);
        for (size_t i = 0; i < residual.size(); i++) {
            residual[i] = static_cast<float>(i) / residual.size();
        }

        auto residual_compressed = alice::residual_compress(residual, 8, 6);
        std::cout << "Original: " << residual.size() * sizeof(float) << " bytes" << std::endl;
        std::cout << "Compressed: " << residual_compressed.size() << " bytes" << std::endl;

        auto residual_decompressed = alice::residual_decompress(residual_compressed);
        std::cout << "Decompressed: " << residual_decompressed.size() << " floats" << std::endl;

        // Calculate error
        float max_error = 0.0f;
        for (size_t i = 0; i < residual.size(); i++) {
            float error = std::abs(residual[i] - residual_decompressed[i]);
            max_error = std::max(max_error, error);
        }
        std::cout << "Max error: " << max_error << std::endl;

    } catch (const alice::Exception& e) {
        std::cerr << "ALICE Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "Done!" << std::endl;
    return 0;
}
