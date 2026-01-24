// Copyright (c) 2024-2026 Moroya Sakamoto. MIT License.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "AliceZipLibrary.generated.h"

/**
 * ALICE-Zip Blueprint Function Library
 *
 * Provides high-performance procedural generation and compression functions
 * for use in Blueprints and C++.
 */
UCLASS()
class ALICEZIP_API UAliceZipLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	// ========================================================================
	// Version Information
	// ========================================================================

	/**
	 * Get the ALICE-Zip library version string
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Version")
	static FString GetVersion();

	/**
	 * Get the ALICE-Zip library version numbers
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Version")
	static void GetVersionNumbers(int32& Major, int32& Minor, int32& Patch);

	// ========================================================================
	// Perlin Noise Generation
	// ========================================================================

	/**
	 * Generate 2D Perlin noise as a float array
	 *
	 * @param Width Width of the noise texture
	 * @param Height Height of the noise texture
	 * @param Seed Random seed for reproducibility
	 * @param Scale Noise scale (larger = more zoomed out)
	 * @param Octaves Number of octaves for fractal noise
	 * @param OutData Output float array (row-major order)
	 * @return True if successful
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Perlin")
	static bool GeneratePerlin2D(
		int32 Width,
		int32 Height,
		int64 Seed,
		float Scale,
		int32 Octaves,
		TArray<float>& OutData);

	/**
	 * Generate advanced 2D Perlin noise with persistence and lacunarity
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Perlin")
	static bool GeneratePerlinAdvanced(
		int32 Width,
		int32 Height,
		int64 Seed,
		float Scale,
		int32 Octaves,
		float Persistence,
		float Lacunarity,
		TArray<float>& OutData);

	/**
	 * Generate a Perlin noise texture
	 *
	 * @param Width Texture width
	 * @param Height Texture height
	 * @param Seed Random seed
	 * @param Scale Noise scale
	 * @param Octaves Number of octaves
	 * @return Generated texture (R8 format)
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Perlin")
	static UTexture2D* GeneratePerlinTexture(
		int32 Width,
		int32 Height,
		int64 Seed = 42,
		float Scale = 10.0f,
		int32 Octaves = 4);

	// ========================================================================
	// Signal Generation
	// ========================================================================

	/**
	 * Generate a sine wave
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Signal")
	static bool GenerateSineWave(
		int32 NumSamples,
		float Frequency,
		float Amplitude,
		float Phase,
		float DCOffset,
		TArray<float>& OutData);

	/**
	 * Generate polynomial data: y = c0 + c1*x + c2*x^2 + ...
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Signal")
	static bool GeneratePolynomial(
		int32 NumSamples,
		const TArray<float>& Coefficients,
		TArray<float>& OutData);

	// ========================================================================
	// Compression Functions
	// ========================================================================

	/**
	 * Compress data using LZMA algorithm
	 *
	 * @param Data Data to compress
	 * @param Preset Compression preset (0-9, higher = better compression)
	 * @param OutCompressed Compressed data
	 * @return True if successful
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Compression")
	static bool LzmaCompress(
		const TArray<uint8>& Data,
		int32 Preset,
		TArray<uint8>& OutCompressed);

	/**
	 * Decompress LZMA data
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Compression")
	static bool LzmaDecompress(
		const TArray<uint8>& CompressedData,
		TArray<uint8>& OutData);

	/**
	 * Compress data using zlib algorithm
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Compression")
	static bool ZlibCompress(
		const TArray<uint8>& Data,
		int32 Level,
		TArray<uint8>& OutCompressed);

	/**
	 * Decompress zlib data
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Compression")
	static bool ZlibDecompress(
		const TArray<uint8>& CompressedData,
		TArray<uint8>& OutData);

	/**
	 * Compress float residuals with quantization (lossy)
	 *
	 * @param Residual Float data to compress
	 * @param Bits Quantization bits (8, 12, or 16)
	 * @param LzmaPreset LZMA compression preset
	 * @param OutCompressed Compressed data
	 * @return True if successful
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Compression")
	static bool ResidualCompress(
		const TArray<float>& Residual,
		int32 Bits,
		int32 LzmaPreset,
		TArray<uint8>& OutCompressed);

	/**
	 * Decompress quantized residuals
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Compression")
	static bool ResidualDecompress(
		const TArray<uint8>& CompressedData,
		TArray<float>& OutResidual);

	// ========================================================================
	// Utility Functions
	// ========================================================================

	/**
	 * Convert float array to grayscale texture
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Utility")
	static UTexture2D* FloatArrayToTexture(
		const TArray<float>& Data,
		int32 Width,
		int32 Height);

	/**
	 * Get the last error message from ALICE-Zip
	 */
	UFUNCTION(BlueprintCallable, Category = "ALICE-Zip|Utility")
	static FString GetLastError();
};
