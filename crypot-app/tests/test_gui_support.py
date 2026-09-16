import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gui_support import build_digest_report, build_analysis_report, build_peer_message


def test_build_digest_report_contains_all_supported_hashes():
    report = build_digest_report("hello")
    assert set(report) == {"sha256", "sha512", "sha3_256", "sha3_512"}
    assert all(len(value) > 0 for value in report.values())


def test_build_analysis_report_exposes_rnn_and_llm_fields():
    report = build_analysis_report("hello")
    assert report["state_dimension"] == 16
    assert "state_norm" in report
    assert "llm" in report
    assert len(report["sha256"]) == 64


def test_build_peer_message_creates_verifiable_message():
    message = build_peer_message("node-a", "research", {"value": 1})
    assert message.sender == "node-a"
    assert message.kind == "research"
    assert message.verify()
