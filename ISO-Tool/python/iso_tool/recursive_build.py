"""Recursive repository inventory, dependency resolution, build and link orchestration."""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from .external_refs import write_dependency_report

SKIP_DIRS={".git",".hg",".svn","node_modules","__pycache__",".venv","venv","build","dist","out","target","bin","obj",".vs",".idea"}
EXTENSIONS={".c":"c",".h":"c-header",".cc":"cpp",".cpp":"cpp",".cxx":"cpp",".hpp":"cpp-header",".hh":"cpp-header",".hxx":"cpp-header",".s":"asm",".asm":"asm",".S":"asm",".rs":"rust",".go":"go",".py":"python",".js":"javascript",".mjs":"javascript",".ts":"typescript",".java":"java",".cs":"csharp",".fs":"fsharp",".fsx":"fsharp"}
MANIFESTS={"CMakeLists.txt":"cmake","Makefile":"make","makefile":"make","configure.ac":"autotools","meson.build":"meson","Cargo.toml":"cargo","go.mod":"go","package.json":"node","pyproject.toml":"python","setup.py":"python","pom.xml":"maven","build.gradle":"gradle","build.gradle.kts":"gradle","*.sln":"msbuild","*.vcxproj":"msbuild","*.csproj":"dotnet"}

@dataclass
class SourceRecord: path:str; language:str; size:int; entry_point:bool=False
@dataclass
class ManifestRecord: path:str; kind:str
@dataclass
class BuildArtifact: path:str; kind:str; command:List[str]=field(default_factory=list); returncode:int=0; status:str="planned"
@dataclass
class RecursiveReport:
    root:str; sources:List[SourceRecord]; manifests:List[ManifestRecord]; artifacts:List[BuildArtifact]; skipped:List[str]; errors:List[str]
    external_references:int=0; unresolved_references:List[str]=field(default_factory=list)

def _manifest_kind(name:str)->Optional[str]:
    if name in MANIFESTS:return MANIFESTS[name]
    for pattern,kind in MANIFESTS.items():
        if pattern.startswith("*") and name.endswith(pattern[1:]):return kind
    return None

def iter_files(root:Path)->Iterable[Path]:
    for base,dirs,files in os.walk(str(root)):
        dirs[:]=[d for d in dirs if d not in SKIP_DIRS]
        for name in files:yield Path(base)/name

def _looks_like_entry(path:Path)->bool:
    if path.suffix.lower() not in {".c",".cc",".cpp",".cxx"}:return False
    try:text=path.read_text(encoding="utf-8",errors="ignore")
    except OSError:return False
    return bool(re.search(r"\b(?:int|auto)\s+main\s*\(",text)) or "wWinMain(" in text)

def inventory(root:Path)->Tuple[List[SourceRecord],List[ManifestRecord]]:
    sources=[]; manifests=[]
    for path in iter_files(root):
        rel=path.relative_to(root).as_posix(); kind=_manifest_kind(path.name)
        if kind:manifests.append(ManifestRecord(rel,kind))
        language=EXTENSIONS.get(path.suffix)
        if language:
            try:size=path.stat().st_size
            except OSError:size=0
            sources.append(SourceRecord(rel,language,size,_looks_like_entry(path)))
    return sorted(sources,key=lambda x:x.path),sorted(manifests,key=lambda x:x.path)

def _tool(name:str)->Optional[str]:return shutil.which(name)

def _run(cmd:Sequence[str],cwd:Path,log:Optional[Path]=None)->int:
    proc=subprocess.run(list(cmd),cwd=str(cwd),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace")
    if log:
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("a",encoding="utf-8") as fh:fh.write("$ "+" ".join(cmd)+"\n"+proc.stdout+"\n")
    return proc.returncode

def acquire_repository(source:str,destination:Optional[Path]=None)->Tuple[Path,Optional[Path]]:
    candidate=Path(source).expanduser()
    if candidate.exists():return candidate.resolve(),None
    parsed=urlparse(source)
    if parsed.scheme not in {"https","http","git"} or not parsed.netloc:raise ValueError("repository must be an existing path or a Git URL")
    git=_tool("git")
    if not git:raise RuntimeError("git is required to acquire a remote repository")
    if destination:destination.mkdir(parents=True,exist_ok=True); checkout,cleanup=destination/"repository",None
    else:
        temp=Path(tempfile.mkdtemp(prefix="iso-tool-repo-")); checkout,cleanup=temp/"repository",temp
    if _run([git,"clone","--recursive","--depth","1",source,str(checkout)],checkout.parent)!=0:
        if cleanup:shutil.rmtree(str(cleanup),ignore_errors=True)
        raise RuntimeError("git clone failed")
    return checkout.resolve(),cleanup

def _pragma_libraries(root:Path,sources:List[SourceRecord])->List[str]:
    pattern=re.compile(r"#\s*pragma\s+comment\s*\(\s*lib\s*,\s*[\"']([^\"']+)[\"']",re.I); libs=[]
    for record in sources:
        if record.language not in {"c","cpp"}:continue
        try:text=(root/record.path).read_text(encoding="utf-8",errors="ignore")
        except OSError:continue
        libs.extend(pattern.findall(text))
    return sorted(set(libs))

def _native_direct_build(root:Path,sources:List[SourceRecord],out:Path,log:Path)->Tuple[List[BuildArtifact],List[str]]:
    native=[s for s in sources if s.language in {"c","cpp"}]
    if not native:return [],[]
    compiler=_tool("c++") or _tool("g++") or _tool("clang++"); ccompiler=_tool("cc") or _tool("gcc") or _tool("clang")
    if not compiler and not ccompiler:return [],["No C/C++ compiler found; native direct-build jobs remain planned."]
    objdir=out/"objects";objdir.mkdir(parents=True,exist_ok=True);artifacts=[];errors=[];objects=[]
    for record in native:
        src=root/record.path;obj=objdir/(record.path.replace("/","__")+".o");obj.parent.mkdir(parents=True,exist_ok=True);cc=ccompiler if record.language=="c" else compiler
        if not cc:errors.append("No compiler for "+record.path);continue
        cmd=[cc,"-c","-O2","-std=c11" if record.language=="c" else "-std=c++17",str(src),"-o",str(obj)];rc=_run(cmd,root,log)
        artifacts.append(BuildArtifact(str(obj.relative_to(out)),"object",cmd,rc,"built" if rc==0 else "failed"))
        if rc==0:objects.append(obj)
        else:errors.append("Compile failed: "+record.path)
    entries=[s for s in native if s.entry_point]
    if len(entries)==1 and objects:
        exe=out/("recursive-native.exe" if os.name=="nt" else "recursive-native");cmd=[compiler or ccompiler]+[str(p) for p in objects]+["-o",str(exe)]
        # GNU/Clang do not consume MSVC pragma comments; translate discovered library names on Windows.
        if compiler and os.name=="nt":
            for lib in _pragma_libraries(root,sources):
                stem=Path(lib).stem
                if stem.lower().endswith(".lib"):stem=stem[:-4]
                if stem.lower().endswith(".dll"):stem=stem[:-4]
                cmd.append("-l"+stem)
        rc=_run(cmd,root,log);artifacts.append(BuildArtifact(str(exe.relative_to(out)),"executable",cmd,rc,"built" if rc==0 else "failed"))
        if rc!=0:errors.append("Native link failed; inspect the recursive build log.")
    elif len(entries)>1:errors.append("Multiple native entry points detected; sources are compiled but not force-linked into one executable.")
    return artifacts,errors

def _resolver_commands(kind:str,work:Path)->List[List[str]]:
    if kind=="node":return [["npm","ci"]] if (work/"package-lock.json").is_file() else [["npm","install"]]
    if kind=="cargo":return [["cargo","fetch"]]
    if kind=="go":return [["go","mod","download"]]
    if kind=="dotnet":return [["dotnet","restore"]]
    if kind=="msbuild":return [["msbuild","/m","/t:Restore"]]
    if kind=="maven":return [["mvn","-B","dependency:go-offline"]]
    if kind=="gradle":return [["gradle","dependencies"]]
    return []

def build_repository(root:Path,output:Optional[Path]=None,execute:bool=False,resolve_dependencies:bool=True)->RecursiveReport:
    root=root.resolve();out=(output or root/"ISO-Tool-build").resolve();out.mkdir(parents=True,exist_ok=True);sources,manifests=inventory(root);artifacts=[];errors=[];skipped=[]
    graph=write_dependency_report(root,out/"external-reference-report.json")
    if not execute:
        artifacts=[BuildArtifact(m.path,"build-manifest",status="planned") for m in manifests]
        return RecursiveReport(str(root),sources,manifests,artifacts,skipped,errors,len(graph.references),graph.unresolved)
    log=out/"recursive-build.log"
    handlers={"cmake":["cmake","--build","build","--config","Release"],"make":["make"],"meson":["meson","compile","-C","build"],"cargo":["cargo","build","--release"],"go":["go","build","./..."],"maven":["mvn","-B","package"],"gradle":["gradle","build"],"msbuild":["msbuild","/m","/p:Configuration=Release"],"dotnet":["dotnet","build","-c","Release"],"node":["npm","run","build"]}
    for manifest in manifests:
        kind=manifest.kind;work=root/Path(manifest.path).parent
        if resolve_dependencies:
            for resolve in _resolver_commands(kind,work):
                if not _tool(resolve[0]):skipped.append(manifest.path+": "+resolve[0]+" not installed");continue
                rc=_run(resolve,work,log);artifacts.append(BuildArtifact(manifest.path,"dependency-resolution",resolve,rc,"resolved" if rc==0 else "failed"))
                if rc!=0:errors.append("Dependency resolution failed: "+manifest.path);break
        command=handlers.get(kind)
        if not command:skipped.append(manifest.path+": no universal native build command");continue
        if not _tool(command[0]):skipped.append(manifest.path+": "+command[0]+" not installed");continue
        if kind=="cmake" and not (work/"build").exists():
            rc=_run(["cmake","-S",str(work),"-B",str(work/"build"),"-DCMAKE_BUILD_TYPE=Release"],root,log)
            if rc!=0:errors.append("CMake configure failed: "+manifest.path);continue
        rc=_run(command,work,log);artifacts.append(BuildArtifact(manifest.path,kind,command,rc,"built" if rc==0 else "failed"))
        if rc!=0:errors.append("Build failed: "+manifest.path)
    direct,direct_errors=_native_direct_build(root,sources,out,log);artifacts.extend(direct);errors.extend(direct_errors)
    python=_tool("python") or _tool("python3")
    if any(s.language=="python" for s in sources) and python:
        cmd=[python,"-m","compileall","-q",str(root)];rc=_run(cmd,root,log);artifacts.append(BuildArtifact("__pycache__","python-bytecode",cmd,rc,"built" if rc==0 else "failed"));
        if rc!=0:errors.append("Python bytecode compilation failed")
    report=RecursiveReport(str(root),sources,manifests,artifacts,skipped,errors,len(graph.references),graph.unresolved)
    (out/"recursive-build-report.json").write_text(json.dumps({"root":report.root,"sources":[asdict(x) for x in report.sources],"manifests":[asdict(x) for x in report.manifests],"artifacts":[asdict(x) for x in report.artifacts],"skipped":report.skipped,"errors":report.errors,"external_references":report.external_references,"unresolved_references":report.unresolved_references},indent=2),encoding="utf-8")
    return report

def main(argv:Optional[Sequence[str]]=None)->int:
    parser=argparse.ArgumentParser(description="Recursively inventory, resolve dependencies, build and link a GitHub/local repository");parser.add_argument("source");parser.add_argument("--output",default=None);parser.add_argument("--execute",action="store_true");parser.add_argument("--no-resolve-dependencies",action="store_true");args=parser.parse_args(argv)
    root,cleanup=acquire_repository(args.source,Path(args.output) if args.output else None)
    try:
        report=build_repository(root,Path(args.output)/"artifacts" if args.output and cleanup is None else None,args.execute,not args.no_resolve_dependencies)
        print(json.dumps({"root":report.root,"source_count":len(report.sources),"manifest_count":len(report.manifests),"artifact_count":len(report.artifacts),"external_reference_count":report.external_references,"errors":report.errors,"skipped":report.skipped},indent=2));return 1 if report.errors else 0
    finally:
        if cleanup:shutil.rmtree(str(cleanup),ignore_errors=True)
if __name__=="__main__":raise SystemExit(main())
