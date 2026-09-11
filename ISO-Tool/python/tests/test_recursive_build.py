import json
import tempfile
import unittest
from pathlib import Path
from iso_tool.external_refs import discover_external_references
from iso_tool.recursive_build import build_repository, inventory
from iso_tool.windows_toolchains import discover

class RecursiveBuildTests(unittest.TestCase):
    def test_inventory_walks_nested_sources_and_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'src'/'nested').mkdir(parents=True)
            (root/'src'/'nested'/'main.cpp').write_text('#include "helper.hpp"\n#pragma comment(lib, "comctl32.lib")\nint main() { return 0; }',encoding='utf-8')
            (root/'src'/'nested'/'helper.hpp').write_text('#pragma once\n',encoding='utf-8');(root/'src'/'nested'/'helper.c').write_text('int helper(void) { return 1; }',encoding='utf-8');(root/'src'/'CMakeLists.txt').write_text('project(test)',encoding='utf-8')
            (root/'.git').mkdir();(root/'.git'/'ignored.cpp').write_text('int main(){}',encoding='utf-8')
            sources,manifests=inventory(root);paths={x.path for x in sources}
            self.assertIn('src/nested/main.cpp',paths);self.assertIn('src/nested/helper.c',paths);self.assertNotIn('.git/ignored.cpp',paths);self.assertEqual([m.path for m in manifests],['src/CMakeLists.txt']);self.assertTrue(next(x for x in sources if x.path.endswith('main.cpp')).entry_point)
    def test_external_reference_graph_resolves_local_include_and_records_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'main.cpp').write_text('#include "local.hpp"\n#pragma comment(lib, "comctl32.lib")\nint main(){}',encoding='utf-8');(root/'local.hpp').write_text('#pragma once\n',encoding='utf-8');graph=discover_external_references(root)
            self.assertTrue(any(r.reference=='local.hpp' and r.resolved for r in graph.references));self.assertTrue(any(r.reference=='comctl32.lib' and r.kind=='link-library' for r in graph.references))
    def test_plan_mode_never_executes_build_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'main.cpp').write_text('int main() { return 0; }',encoding='utf-8');report=build_repository(root,execute=False);self.assertEqual(len(report.sources),1);self.assertTrue(all(a.status=='planned' for a in report.artifacts));self.assertEqual(report.external_references,0)
    def test_explicit_toolchain_selection_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'build';(root/'main.cpp').write_text('int main() { return 0; }',encoding='utf-8');report=build_repository(root,out,execute=False,cpp_compiler='C:/toolchain/g++.exe',linker='C:/toolchain/ld.exe',assembler='C:/toolchain/nasm.exe');selection=json.loads((out/'toolchain-selection.json').read_text(encoding='utf-8'));self.assertEqual(selection['cpp_compiler'],'C:/toolchain/g++.exe');self.assertEqual(selection['linker'],'C:/toolchain/ld.exe');self.assertEqual(selection['assembler'],'C:/toolchain/nasm.exe');self.assertEqual(len(report.sources),1)
    def test_toolchain_detector_returns_a_list(self):
        self.assertIsInstance(discover(),list)

if __name__=='__main__':unittest.main()
