"""Explicit workflow entry points for compiled artifacts and ISO creation."""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class WorkflowEntryPoint:
    id: str
    label: str
    description: str
    handler: str

ENTRY_POINTS = (
    WorkflowEntryPoint("analyze-source", "Analyze source", "Inventory a local checkout or acquired repository.", "analyze_source"),
    WorkflowEntryPoint("build-compiled-images", "Build compiled images", "Compile/assemble/link authorized source jobs and collect artifacts.", "build_compiled_images"),
    WorkflowEntryPoint("import-boot-image", "Import boot sectors / ISO", "Inspect and stage bounded boot artifacts from a local image.", "import_boot_image"),
    WorkflowEntryPoint("inspect-iso", "Advanced ISO inspection", "Read-only ISO 9660, UDF, MBR/GPT and El Torito analysis.", "inspect_iso"),
    WorkflowEntryPoint("build-iso", "Build ISO / IMG", "Stage artifacts and invoke the selected ISO/image backend.", "build_iso"),
    WorkflowEntryPoint("validate-image", "Validate image", "Check output metadata, hashes and configured boot artifacts.", "validate_image"),
)

def list_entry_points():
    return [entry.__dict__ for entry in ENTRY_POINTS]
