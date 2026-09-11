"""End-to-end recursive repository build and user-selected ISO mastering."""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

# Permit both `python -m iso_tool...` and direct execution from ISO-Tool/python.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from iso_tool.recursive_build import acquire_repository, build_repository
    from iso_tool.image import create_iso
else:
    from .iso_tool.recursive_build import acquire_repository, build_repository
    from .iso_tool.image import create_iso


def stage_repository(root: Path, build_output: Path) -> Path:
    staging=build_output/"iso-staging"
    if staging.exists(): shutil.rmtree(str(staging))
    staging.mkdir(parents=True)
    excluded={".git",".github","ISO-Tool-build","node_modules","__pycache__",".venv","venv","build","dist","target","obj","bin"}
    for source in root.rglob("*"):
        rel=source.relative_to(root)
        if any(part in excluded for part in rel.parts): continue
        target=staging/rel
        if source.is_dir(): target.mkdir(parents=True,exist_ok=True)
        elif source.is_file(): target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(source),str(target))
    artifact_root=build_output/"artifacts"
    if artifact_root.exists(): shutil.copytree(str(artifact_root),str(staging/"compiled-artifacts"),dirs_exist_ok=True)
    return staging


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description="Recursively build a repository and create an ISO at the user-selected path")
    parser.add_argument("source",help="local repository or Git URL");parser.add_argument("--output",required=True,help="final ISO path, including desired filename");parser.add_argument("--label",default="ISO_TOOL");parser.add_argument("--profile",default="data");parser.add_argument("--build-output",default=None);parser.add_argument("--no-resolve-dependencies",action="store_true");args=parser.parse_args(argv)
    output=Path(args.output).expanduser().resolve();build_output=Path(args.build_output).expanduser().resolve() if args.build_output else output.parent/(output.stem+".iso-tool-build")
    root,cleanup=acquire_repository(args.source,build_output/"source")
    try:
        report=build_repository(root,build_output/"artifacts",True,not args.no_resolve_dependencies)
        if report.errors: raise RuntimeError("recursive build reported errors; see recursive-build-report.json")
        staging=stage_repository(root,build_output);result=create_iso(staging,output,args.label,args.profile)
        print(json.dumps({"output":str(output),"backend_returncode":result.returncode,"source_count":len(report.sources),"artifact_count":len(report.artifacts),"external_reference_count":report.external_references,"staging":str(staging)},indent=2));return 0
    finally:
        if cleanup:shutil.rmtree(str(cleanup),ignore_errors=True)

if __name__=="__main__": raise SystemExit(main())
