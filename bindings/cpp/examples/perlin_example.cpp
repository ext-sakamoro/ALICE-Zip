/**
 * ALICE-Zip C++ Example: Perlin Noise Generation
 *
 * This example demonstrates how to generate Perlin noise textures
 * using the ALICE-Zip C++ bindings.
 */

#include "alice.hpp"
#include <iostream>
#include <fstream>

int main() {
    std::cout << "ALICE-Zip C++ Perlin Noise Example" << std::endl;
    std::cout << "Version: " << alice::version() << std::endl;
    std::cout << std::endl;

    // Generate 2D Perlin noise
    const size_t width = 256;
    const size_t height = 256;
    const uint64_t seed = 42;
    const float scale = 10.0f;
    const uint32_t octaves = 4;

    std::cout << "Generating " << width << "x" << height << " Perlin noise..." << std::endl;

    try {
        auto noise = alice::perlin_2d(width, height, seed, scale, octaves);

        std::cout << "Generated " << noise.size() << " noise values" << std::endl;

        // Calculate statistics
        float min_val = noise[0];
        float max_val = noise[0];
        float sum = 0.0f;

        for (float v : noise) {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
            sum += v;
        }

        float avg = sum / noise.size();

        std::cout << "Min: " << min_val << std::endl;
        std::cout << "Max: " << max_val << std::endl;
        std::cout << "Avg: " << avg << std::endl;

        // Save as raw grayscale image (can be viewed with ImageMagick: display -size 256x256 -depth 8 gray:perlin.raw)
        std::ofstream file("perlin.raw", std::ios::binary);
        if (file.is_open()) {
            for (float v : noise) {
                uint8_t byte = static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f);
                file.write(reinterpret_cast<const char*>(&byte), 1);
            }
            file.close();
            std::cout << "Saved to perlin.raw" << std::endl;
        }

        // Generate advanced Perlin noise with custom persistence/lacunarity
        std::cout << std::endl;
        std::cout << "Generating advanced Perlin noise..." << std::endl;

        auto advanced = alice::perlin_advanced(
            width, height, seed, scale, octaves,
            0.5f,  // persistence
            2.0f   // lacunarity
        );

        std::cout << "Generated " << advanced.size() << " advanced noise values" << std::endl;

    } catch (const alice::Exception& e) {
        std::cerr << "ALICE Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "Done!" << std::endl;
    return 0;
}
