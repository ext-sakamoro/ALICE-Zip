// Copyright (c) 2024-2026 Moroya Sakamoto. MIT License.

using UnrealBuildTool;
using System.IO;

public class AliceZip : ModuleRules
{
	public AliceZip(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[] {
				Path.Combine(ModuleDirectory, "Public"),
				Path.Combine(PluginDirectory, "ThirdParty", "include")
			}
		);

		PrivateIncludePaths.AddRange(
			new string[] {
				Path.Combine(ModuleDirectory, "Private")
			}
		);

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine"
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"Projects"
			}
		);

		// Link native library
		string LibPath = Path.Combine(PluginDirectory, "ThirdParty", "lib");

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			PublicAdditionalLibraries.Add(Path.Combine(LibPath, "Win64", "alice_core.lib"));
			RuntimeDependencies.Add(Path.Combine(LibPath, "Win64", "alice_core.dll"));
			PublicDelayLoadDLLs.Add("alice_core.dll");
		}
		else if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			PublicAdditionalLibraries.Add(Path.Combine(LibPath, "Mac", "libalice_core.dylib"));
			RuntimeDependencies.Add(Path.Combine(LibPath, "Mac", "libalice_core.dylib"));
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			PublicAdditionalLibraries.Add(Path.Combine(LibPath, "Linux", "libalice_core.so"));
			RuntimeDependencies.Add(Path.Combine(LibPath, "Linux", "libalice_core.so"));
		}

		// Allow unsafe code for FFI
		bEnableUndefinedIdentifierWarnings = false;
	}
}
