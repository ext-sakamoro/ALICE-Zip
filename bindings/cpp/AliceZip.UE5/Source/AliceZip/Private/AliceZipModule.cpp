// Copyright (c) 2024-2026 Moroya Sakamoto. MIT License.

#include "AliceZipModule.h"
#include "Core.h"
#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

#define LOCTEXT_NAMESPACE "FAliceZipModule"

void FAliceZipModule::StartupModule()
{
	// Get the base directory of this plugin
	FString BaseDir = IPluginManager::Get().FindPlugin("AliceZip")->GetBaseDir();

	// Build the path to the native library
	FString LibraryPath;
#if PLATFORM_WINDOWS
	LibraryPath = FPaths::Combine(*BaseDir, TEXT("ThirdParty/lib/Win64/alice_core.dll"));
#elif PLATFORM_MAC
	LibraryPath = FPaths::Combine(*BaseDir, TEXT("ThirdParty/lib/Mac/libalice_core.dylib"));
#elif PLATFORM_LINUX
	LibraryPath = FPaths::Combine(*BaseDir, TEXT("ThirdParty/lib/Linux/libalice_core.so"));
#endif

	// Load the DLL
	LibraryHandle = !LibraryPath.IsEmpty() ? FPlatformProcess::GetDllHandle(*LibraryPath) : nullptr;

	if (LibraryHandle)
	{
		UE_LOG(LogTemp, Log, TEXT("ALICE-Zip: Native library loaded successfully"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("ALICE-Zip: Failed to load native library from %s"), *LibraryPath);
	}
}

void FAliceZipModule::ShutdownModule()
{
	// Free the DLL handle
	if (LibraryHandle)
	{
		FPlatformProcess::FreeDllHandle(LibraryHandle);
		LibraryHandle = nullptr;
	}
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FAliceZipModule, AliceZip)
