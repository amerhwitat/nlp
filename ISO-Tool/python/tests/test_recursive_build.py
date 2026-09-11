import tempfile
import unittest
from pathlib import Path

from iso_tool.recursive_build import build_repository, inventory


class RecursiveBuildTests(unittest.TestCase):
    def test_inventory_walks_nested_sources_and_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "nested").mkdir(parents=True)
            (root / "src" / "nested" / "main.cpp").write_text("int main() { return 0; }", encoding="utf-8")
            (root / "src" / "nested" / "helper.c").write_text("int helper(void) { return 1; }", encoding="utf-8")
            (root / "src" / "CMakeLists.txt").write_text("project(test)", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "ignored.cpp").write_text("int main(){}", encoding="utf-8")

            sources, manifests = inventory(root)
            paths = {item.path for item in sources}
            self.assertIn("src/nested/main.cpp", paths)
            self.assertIn("src/nested/helper.c", paths)
            self.assertNotIn(".git/ignored.cpp", paths)
            self.assertEqual([m.path for m in manifests], ["src/CMakeLists.txt"])
            self.assertTrue(next(x for x in sources if x.path.endswith("main.cpp")).entry_point)

    def test_plan_mode_never_executes_build_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.cpp").write_text("int main() { return 0; }", encoding="utf-8")
            report = build_repository(root, execute=False)
            self.assertEqual(len(report.sources), 1)
            self.assertTrue(all(a.status == "planned" for a in report.artifacts))


if __name__ == "__main__":
    unittest.main()
