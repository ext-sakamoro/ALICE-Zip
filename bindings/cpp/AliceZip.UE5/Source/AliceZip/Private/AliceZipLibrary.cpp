// Copyright (c) 2024-2026 Moroya Sakamoto. MIT License.

#include "AliceZipLibrary.h"
#include "Engine/Texture2D.h"

// Include the C header
extern "C" {
#include "alice.h"
}

// ============================================================================
// Version Information
// ============================================================================

FString UAliceZipLibrary::GetVersion()
{
	const char* Version = alice_version();
	return FString(UTF8_TO_TCHAR(Version));
}

void UAliceZipLibrary::GetVersionNumbers(int32& Major, int32& Minor, int32& Patch)
{
	uint32 M, N, P;
	alice_version_numbers(&M, &N, &P);
	Major = static_cast<int32>(M);
	Minor = static_cast<int32>(N);
	Patch = static_cast<int32>(P);
}

FString UAliceZipLibrary::GetLastError()
{
	const char* Error = alice_get_last_error();
	return Error ? FString(UTF8_TO_TCHAR(Error)) : FString();
}

// ============================================================================
// Perlin Noise Generation
// ============================================================================

bool UAliceZipLibrary::GeneratePerlin2D(
	int32 Width,
	int32 Height,
	int64 Seed,
	float Scale,
	int32 Octaves,
	TArray<float>& OutData)
{
	if (Width <= 0 || Height <= 0)
	{
		return false;
	}

	AliceFloatBuffer Buffer = {nullptr, 0, 0};
	AliceError Error = alice_perlin_2d(
		static_cast<size_t>(Width),
		static_cast<size_t>(Height),
		static_cast<uint64_t>(Seed),
		Scale,
		static_cast<uint32_t>(Octaves),
		&Buffer);

	if (Error != ALICE_SUCCESS)
	{
		return false;
	}

	OutData.SetNumUninitialized(static_cast<int32>(Buffer.len));
	FMemory::Memcpy(OutData.GetData(), Buffer.data, Buffer.len * sizeof(float));

	alice_free_float_buffer(&Buffer);
	return true;
}

bool UAliceZipLibrary::GeneratePerlinAdvanced(
	int32 Width,
	int32 Height,
	int64 Seed,
	float Scale,
	int32 Octaves,
	float Persistence,
	float Lacunarity,
	TArray<float>& OutData)
{
	if (Width <= 0 || Height <= 0)
	{
		return false;
	}

	AliceFloatBuffer Buffer = {nullptr, 0, 0};
	AliceError Error = alice_perlin_advanced(
		static_cast<size_t>(Width),
		static_cast<size_t>(Height),
		static_cast<uint64_t>(Seed),
		Scale,
		static_cast<uint32_t>(Octaves),
		Persistence,
		Lacunarity,
		&Buffer);

	if (Error != ALICE_SUCCESS)
	{
		return false;
	}

	OutData.SetNumUninitialized(static_cast<int32>(Buffer.len));
	FMemory::Memcpy(OutData.GetData(), Buffer.data, Buffer.len * sizeof(float));

	alice_free_float_buffer(&Buffer);
	return true;
}

UTexture2D* UAliceZipLibrary::GeneratePerlinTexture(
	int32 Width,
	int32 Height,
	int64 Seed,
	float Scale,
	int32 Octaves)
{
	TArray<float> NoiseData;
	if (!GeneratePerlin2D(Width, Height, Seed, Scale, Octaves, NoiseData))
	{
		return nullptr;
	}

	return FloatArrayToTexture(NoiseData, Width, Height);
}

// ============================================================================
// Signal Generation
// ============================================================================

bool UAliceZipLibrary::GenerateSineWave(
	int32 NumSamples,
	float Frequency,
	float Amplitude,
	float Phase,
	float DCOffset,
	TArray<float>& OutData)
{
	if (NumSamples <= 0)
	{
		return false;
	}

	AliceFloatBuffer Buffer = {nullptr, 0, 0};
	AliceError Error = alice_sine_wave(
		static_cast<size_t>(NumSamples),
		Frequency,
		Amplitude,
		Phase,
		DCOffset,
		&Buffer);

	if (Error != ALICE_SUCCESS)
	{
		return false;
	}

	OutData.SetNumUninitialized(static_cast<int32>(Buffer.len));
	FMemory::Memcpy(OutData.GetData(), Buffer.data, Buffer.len * sizeof(float));

	alice_free_float_buffer(&Buffer);
	return true;
}

bool UAliceZipLibrary::GeneratePolynomial(
	int32 NumSamples,
	const TArray<float>& Coefficients,
	TArray<float>& OutData)
{
	if (NumSamples <= 0 || Coefficients.Num() == 0)
	{
		return false;
	}

	// Convert float coefficients to double
	TArray<double> DoubleCoeffs;
	DoubleCoeffs.SetNumUninitialized(Coefficients.Num());
	for (int32 i = 0; i < Coefficients.Num(); i++)
	{
		DoubleCoeffs[i] = static_cast<double>(Coefficients[i]);
	}

	AliceFloatBuffer Buffer = {nullptr, 0, 0};
	AliceError Error = alice_polynomial_generate(
		static_cast<size_t>(NumSamples),
		DoubleCoeffs.GetData(),
		static_cast<size_t>(DoubleCoeffs.Num()),
		&Buffer);

	if (Error != ALICE_SUCCESS)
	{
		return false;
	}

	OutData.SetNumUninitialized(static_cast<int32>(Buffer.len));
	FMemory::Memcpy(OutData.GetData(), Buffer.data, Buffer.len * sizeof(float));

	alice_free_float_buffer(&Buffer);
	return true;
}

// ============================================================================
// Compression Functions
// ============================================================================

bool UAliceZipLibrary::LzmaCompress(
	const TArray<uint8>& Data,
	int32 Preset,
	TArray<uint8>& OutCompressed)
{
	if (Data.Num() == 0)
	{
		return false;
	}

	AliceBuffer Buffer = {nullptr, 0, 0};
	AliceError Error = alice_lzma_compress(
		Data.GetData(),
		static_cast<size_t>(Data.Num()),
		static_cast<uint32_t>(FMath::Clamp(Preset, 0, 9)),
		&Buffer);

	if (Error != ALICE_SUCCESS)
	{
		return false;
	}

	OutCompressed.SetNumUninitialized(static_cast<int32>(Buffer.len));
	FMemory::Memcpy(OutCompressed.GetData(), Buffer.data, Buffer.len);

	alice_free_buffer(&Buffer);
	return true;
}

bool UAliceZipLibrary::LzmaDecompress(
	const TArray<uint8>& CompressedData,
	TArray<uint8>& OutData)
{
	if (CompressedData.Num() == 0)
	{
		return false;
	}

	AliceBuffer Buffer = {nullptr, 0, 0};
	AliceError Error = alice_lzma_decompress(
		CompressedData.GetData(),
		static_cast<size_t>(CompressedData.Num()),
		&Buffer);

	if (Error != ALICE_SUCCESS)
	{
		return false;
	}

	OutData.SetNumUninitialized(static_cast<int32>(Buffer.len));
	FMemory::Memcpy(OutData.GetData(), Buffer.data, Buffer.len);

	alice_free_buffer(&Buffer);
	return true;
}

bool UAliceZipLibrary::ZlibCompress(
	const TArray<uint8>& Data,
	int32 Level,
	TArray<uint8>& OutCompressed)
{
	if (Data.Num() == 0)
	{
		return false;
	}

	AliceBuffer Buffer = {nullptr, 0, 0};
	AliceError Error = alice_zlib_compress(
		Data.GetData(),
		static_cast<size_t>(Data.Num()),
		static_cast<uint32_t>(FMath::Clamp(Level, 1, 9)),
		&Buffer);

	if (Error != ALICE_SUCCESS)
	{
		return false;
	}

	OutCompressed.SetNumUninitialized(static_cast<int32>(Buffer.len));
	FMemory::Memcpy(OutCompressed.GetData(), Buffer.data, Buffer.len);

	alice_free_buffer(&Buffer);
	return true;
}

bool UAliceZipLibrary::ZlibDecompress(
	const TArray<uint8>& CompressedData,
	TArray<uint8>& OutData)
{
	if (CompressedData.Num() == 0)
	{
		return false;
	}

	AliceBuffer Buffer = {nullptr, 0, 0};
	AliceError Error = alice_zlib_decompress(
		CompressedData.GetData(),
		static_cast<size_t>(CompressedData.Num()),
		&Buffer);

	if (Error != ALICE_SUCCESS)
	{
		return false;
	}

	OutData.SetNumUninitialized(static_cast<int32>(Buffer.len));
	FMemory::Memcpy(OutData.GetData(), Buffer.data, Buffer.len);

	alice_free_buffer(&Buffer);
	return true;
}

bool UAliceZipLibrary::ResidualCompress(
	const TArray<float>& Residual,
	int32 Bits,
	int32 LzmaPreset,
	TArray<uint8>& OutCompressed)
{
	if (Residual.Num() == 0)
	{
		return false;
	}

	AliceBuffer Buffer = {nullptr, 0, 0};
	AliceError Error = alice_residual_compress(
		Residual.GetData(),
		static_cast<size_t>(Residual.Num()),
		static_cast<uint8_t>(FMath::Clamp(Bits, 8, 16)),
		static_cast<uint32_t>(FMath::Clamp(LzmaPreset, 0, 9)),
		&Buffer);

	if (Error != ALICE_SUCCESS)
	{
		return false;
	}

	OutCompressed.SetNumUninitialized(static_cast<int32>(Buffer.len));
	FMemory::Memcpy(OutCompressed.GetData(), Buffer.data, Buffer.len);

	alice_free_buffer(&Buffer);
	return true;
}

bool UAliceZipLibrary::ResidualDecompress(
	const TArray<uint8>& CompressedData,
	TArray<float>& OutResidual)
{
	if (CompressedData.Num() == 0)
	{
		return false;
	}

	AliceFloatBuffer Buffer = {nullptr, 0, 0};
	AliceError Error = alice_residual_decompress(
		CompressedData.GetData(),
		static_cast<size_t>(CompressedData.Num()),
		&Buffer);

	if (Error != ALICE_SUCCESS)
	{
		return false;
	}

	OutResidual.SetNumUninitialized(static_cast<int32>(Buffer.len));
	FMemory::Memcpy(OutResidual.GetData(), Buffer.data, Buffer.len * sizeof(float));

	alice_free_float_buffer(&Buffer);
	return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

UTexture2D* UAliceZipLibrary::FloatArrayToTexture(
	const TArray<float>& Data,
	int32 Width,
	int32 Height)
{
	if (Data.Num() != Width * Height || Width <= 0 || Height <= 0)
	{
		return nullptr;
	}

	UTexture2D* Texture = UTexture2D::CreateTransient(Width, Height, PF_G8);
	if (!Texture)
	{
		return nullptr;
	}

	// Lock the texture for writing
	uint8* MipData = static_cast<uint8*>(Texture->GetPlatformData()->Mips[0].BulkData.Lock(LOCK_READ_WRITE));

	// Convert float data to bytes
	for (int32 i = 0; i < Data.Num(); i++)
	{
		MipData[i] = static_cast<uint8>(FMath::Clamp(Data[i], 0.0f, 1.0f) * 255.0f);
	}

	Texture->GetPlatformData()->Mips[0].BulkData.Unlock();
	Texture->UpdateResource();

	return Texture;
}
