import os
from pathlib import Path
from iso_tool.toolchain_detector import ToolSpec, detect_tools, apply_user_environment


def test_detector_finds_tool_on_path(monkeypatch, tmp_path):
    tool = tmp_path / ("fake.exe" if os.name == "nt" else "fake")
    tool.write_text("x", encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    report = detect_tools((ToolSpec("Fake", (), tool.name, "FAKE_HOME"),))
    item = report["tools"][0]
    assert item["status"] == "found"
    assert Path(item["executable_path"]) == tool.resolve()
    assert item["source"] == "PATH"


def test_apply_environment_is_process_local_by_default(monkeypatch, tmp_path):
    tool = tmp_path / ("fake.exe" if os.name == "nt" else "fake")
    tool.write_text("x", encoding="utf-8")
    tool.chmod(0o755)
    report = {"tools": [{"status": "found", "home": str(tmp_path), "env_var": "FAKE_HOME", "executable_path": str(tool)}]}
    result = apply_user_environment(report, persist=False)
    assert result["persisted"] is False
    assert os.environ["FAKE_HOME"] == str(tmp_path)
    assert str(tmp_path) in result["new_paths"]
