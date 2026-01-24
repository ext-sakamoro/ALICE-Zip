// Copyright (c) 2024-2026 Moroya Sakamoto. MIT License.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FAliceZipModule : public IModuleInterface
{
public:
	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

	/**
	 * Singleton-like access to this module's interface.
	 *
	 * @return Returns singleton instance, loading the module on demand if needed
	 */
	static inline FAliceZipModule& Get()
	{
		return FModuleManager::LoadModuleChecked<FAliceZipModule>("AliceZip");
	}

	/**
	 * Checks to see if this module is loaded and ready.
	 *
	 * @return True if the module is loaded and ready to use
	 */
	static inline bool IsAvailable()
	{
		return FModuleManager::Get().IsModuleLoaded("AliceZip");
	}

private:
	/** Handle to the loaded DLL */
	void* LibraryHandle;
};
