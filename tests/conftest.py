"""Shared pytest fixtures.

Every test runs against a fresh temporary DATA_ROOT. We monkeypatch
CONFIG at module load so all downstream imports see the temp paths.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every test's data root, then reload the modules that
    baked CONFIG.data_root at import time. This is coarser than
    dependency-injecting CONFIG everywhere but keeps the production code
    simple: production sets env once at startup and never rebinds.
    """
    data = tmp_path / "capture"
    (data / "inbox" / "audio").mkdir(parents=True)
    (data / "inbox" / "image").mkdir(parents=True)
    (data / "inbox" / "pdf").mkdir(parents=True)
    (data / "processed").mkdir(parents=True)
    (data / "queue" / "classify").mkdir(parents=True)
    (data / "queue" / "embed").mkdir(parents=True)

    monkeypatch.setenv("DATA_ROOT", str(data))
    monkeypatch.setenv("LIBRARY_DB", str(data / "library.db"))
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    # Reload every capture module that reads CONFIG at import time.
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]

    return data
