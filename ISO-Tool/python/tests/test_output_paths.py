from pathlib import Path

from iso_tool.output_paths import prepare_output_layout


def test_prepare_output_layout(tmp_path: Path):
    paths = prepare_output_layout(tmp_path / "chosen")
    assert paths["root"] == (tmp_path / "chosen").resolve()
    for key in ("iso", "boot_images", "binaries", "executables", "libraries", "logs", "manifests"):
        assert paths[key].is_dir()
    assert paths["dependency_cache"].is_dir()
