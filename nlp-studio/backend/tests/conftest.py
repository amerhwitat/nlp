"""Test fixtures for the NLP / Ancient Script Studio backend.

Runtime data is isolated to a fresh temp directory per test session so tests never
touch the developer's uploads/results/history. ``python/thamudic`` is reused via
sys.path, mirroring how the app boots.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Resolve the repository layout before importing anything from the backend.
_TESTS_DIR = Path(__file__).resolve().parent          # backend/tests
_STUDIO_DIR = _TESTS_DIR.parent.parent                # nlp-studio
REPO_ROOT = _STUDIO_DIR.parent                        # repository root (contains python/)

for _path in (str(REPO_ROOT), str(REPO_ROOT / "python"), str(_STUDIO_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Isolate runtime data (uploads/results/translation log) into a temp directory.
_TMP = tempfile.mkdtemp(prefix="nlp-studio-tests-")
os.environ["NLP_STUDIO_DATA_DIR"] = str(Path(_TMP) / "data")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend import config  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture()
def client():
    config.ensure_directories()
    with TestClient(app) as test_client:
        yield test_client
