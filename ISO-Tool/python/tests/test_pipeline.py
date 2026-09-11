import tempfile, unittest
from pathlib import Path
from iso_tool.pipeline import BuildPipeline
from iso_tool.buildplan import make_inventory

class PipelineTests(unittest.TestCase):
    def test_inventory_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'a.cpp').write_text('int main(){}',encoding='utf-8'); (root/'b.asm').write_text('nop',encoding='utf-8')
            p=BuildPipeline(root,workers=2)
            self.assertEqual({x.suffix for x in p.inventory()},{'.cpp','.asm'})
            self.assertEqual(len(p.sha256(root/'a.cpp')),64)
    def test_build_groups(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'a.c').touch(); (root/'b.cpp').touch(); (root/'boot.asm').touch(); (root/'x.cs').touch()
            g=make_inventory(root)
            self.assertEqual(len(g['c']),1); self.assertEqual(len(g['cpp']),1); self.assertEqual(len(g['asm']),1); self.assertEqual(len(g['csharp']),1); self.assertEqual(len(g['boot']),1)

if __name__=='__main__': unittest.main()
