import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from iso_tool.workspace_builder import _safe_id, load_profiles, build_workspace


class WorkspaceBuilderTests(unittest.TestCase):
    def test_safe_id(self):
        self.assertEqual(_safe_id('BizXtreme'), 'bizxtreme')
        self.assertEqual(_safe_id('Chimera II OS'), 'chimera-ii-os')

    def test_load_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'profiles.json'
            path.write_text(json.dumps({'repositories': [{'id': 'demo', 'url': 'https://example.invalid/demo'}]}), encoding='utf-8')
            self.assertEqual(load_profiles(path)['repositories'][0]['id'], 'demo')

    @patch('iso_tool.workspace_builder.create_iso')
    @patch('iso_tool.workspace_builder.build_spit_fire')
    @patch('iso_tool.workspace_builder.build')
    @patch('iso_tool.workspace_builder.prepare_source')
    def test_workspace_build_stages_repository(self, prepare, build, boot, create_iso):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / 'source'
            source.mkdir()
            (source / 'README.md').write_text('demo', encoding='utf-8')
            prepare.return_value = source
            result = {
                'layout': {
                    'executables': tmp / 'repo-out' / 'executables',
                    'libraries': tmp / 'repo-out' / 'libraries',
                    'boot_images': tmp / 'repo-out' / 'boot-images',
                },
                'manifests': tmp / 'manifest.json',
            }
            for key in ('executables', 'libraries', 'boot_images'):
                result['layout'][key].mkdir(parents=True)
            build.return_value = result
            boot.side_effect = lambda source_path, output_path, log=print: output_path.write_bytes(b'boot')
            create_iso.side_effect = lambda staging, output, **kwargs: output.write_bytes(b'iso')
            out = tmp / 'output'
            result = build_workspace({'demo': 'https://example.invalid/demo'}, out)
            self.assertEqual(Path(result['iso']).read_bytes(), b'iso')
            self.assertEqual((out / 'staging' / 'src' / 'demo' / 'README.md').read_text(encoding='utf-8'), 'demo')
            self.assertTrue(Path(result['manifest']).is_file())


if __name__ == '__main__':
    unittest.main()
