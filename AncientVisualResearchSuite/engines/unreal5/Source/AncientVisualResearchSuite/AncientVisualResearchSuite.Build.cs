using UnrealBuildTool;
public class AncientVisualResearchSuite : ModuleRules
{
    public AncientVisualResearchSuite(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] { "Core", "CoreUObject", "Engine", "RenderCore", "RHI" });
    }
}
