"""AST-based source parity generator for ISO-Tool Python modules.

It inventories every Python module and emits deterministic C, C++, C# and Java
parity units containing source identity, imports, classes and function symbols.
Dynamic Python semantics are never silently guessed; the manifest records the
source hash so native implementations can be completed and verified against it.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, re
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class ModuleInfo:
    source: str
    module: str
    sha256: str
    imports: tuple[str, ...]
    classes: tuple[str, ...]
    functions: tuple[str, ...]

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s).strip("_") or "module"

def _cstr(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)

def inspect_file(path: Path, root: Path) -> ModuleInfo:
    raw = path.read_bytes()
    tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    imports, classes, functions = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom): imports.append((node.module or "") + ":" + ",".join(a.name for a in node.names))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): functions.append(node.name)
        elif isinstance(node, ast.ClassDef): classes.append(node.name)
    return ModuleInfo(path.relative_to(root).as_posix(), _safe(path.stem), hashlib.sha256(raw).hexdigest(), tuple(sorted(set(imports))), tuple(sorted(set(classes))), tuple(sorted(set(functions))))

def collect(root: Path):
    return [inspect_file(p, root) for p in sorted(root.rglob("*.py")) if ".git" not in p.parts and "__pycache__" not in p.parts]

def _arr_cpp(xs): return ", ".join(_cstr(x) for x in xs)
def _arr_java(xs): return ", ".join(_cstr(x) for x in xs)

def emit_c(m: ModuleInfo) -> str:
    n=_safe(m.module)
    return f'''/* GENERATED; source: {m.source}; sha256: {m.sha256} */
#include <stddef.h>
static const char* {n}_source = {_cstr(m.source)};
static const char* {n}_sha256 = {_cstr(m.sha256)};
static const char* {n}_imports[] = {{{_arr_cpp(m.imports)}}};
static const char* {n}_classes[] = {{{_arr_cpp(m.classes)}}};
static const char* {n}_functions[] = {{{_arr_cpp(m.functions)}}};
const char* iso_tool_{n}_source(void) {{ return {n}_source; }}
const char* iso_tool_{n}_sha256(void) {{ return {n}_sha256; }}
size_t iso_tool_{n}_function_count(void) {{ return sizeof({n}_functions)/sizeof({n}_functions[0]); }}
'''

def emit_cpp(m: ModuleInfo) -> str:
    n=_safe(m.module)
    return f'''// GENERATED; source: {m.source}; sha256: {m.sha256}
#include <string>
#include <vector>
namespace iso_tool::python_parity {{
struct {n} {{
 static constexpr const char* source={_cstr(m.source)};
 static constexpr const char* sha256={_cstr(m.sha256)};
 static std::vector<std::string> imports() {{ return {{{_arr_cpp(m.imports)}}}; }}
 static std::vector<std::string> classes() {{ return {{{_arr_cpp(m.classes)}}}; }}
 static std::vector<std::string> functions() {{ return {{{_arr_cpp(m.functions)}}}; }}
}};
}}
'''

def emit_csharp(m: ModuleInfo) -> str:
    n=_safe(m.module)
    def arr(xs): return ", ".join(_cstr(x) for x in xs)
    return f'''// GENERATED; source: {m.source}; sha256: {m.sha256}
namespace IsoTool.PythonParity {{
public static class {n} {{
 public const string Source={_cstr(m.source)}; public const string Sha256={_cstr(m.sha256)};
 public static readonly string[] Imports = new string[] {{{arr(m.imports)}}};
 public static readonly string[] Classes = new string[] {{{arr(m.classes)}}};
 public static readonly string[] Functions = new string[] {{{arr(m.functions)}}};
}}
}}
'''

def emit_java(m: ModuleInfo) -> str:
    n=_safe(m.module)
    return f'''// GENERATED; source: {m.source}; sha256: {m.sha256}
package iso.tool.parity.generated;
public final class {n} {{
 public static final String SOURCE={_cstr(m.source)}; public static final String SHA256={_cstr(m.sha256)};
 public static final String[] IMPORTS={{ {_arr_java(m.imports)} }};
 public static final String[] CLASSES={{ {_arr_java(m.classes)} }};
 public static final String[] FUNCTIONS={{ {_arr_java(m.functions)} }};
 private {n}() {{}}
}}
'''

def translate(root: Path, out: Path):
    modules=collect(root)
    for lang in ("c","cpp","csharp","java"): (out/lang).mkdir(parents=True,exist_ok=True)
    for m in modules:
        n=_safe(m.module)
        (out/"c"/f"{n}.c").write_text(emit_c(m),encoding="utf-8")
        (out/"cpp"/f"{n}.cpp").write_text(emit_cpp(m),encoding="utf-8")
        (out/"csharp"/f"{n}.cs").write_text(emit_csharp(m),encoding="utf-8")
        (out/"java"/f"{n}.java").write_text(emit_java(m),encoding="utf-8")
    manifest={"schema":1,"source_root":str(root),"translator":"AST-signature-parity","modules":[asdict(m) for m in modules],"targets":["c","cpp","csharp-net6","csharp-netframework48","java8+"]}
    (out/"parity-manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest

def main(argv=None):
    ap=argparse.ArgumentParser(description="Generate C/C++/C#/.NET/Java parity units for all ISO-Tool Python modules")
    ap.add_argument("root",type=Path); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(argv); print(json.dumps(translate(a.root,a.output),indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
