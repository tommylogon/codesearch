"""Pytest isolation fixtures so tests never touch the user's machine index registry."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_codesearch_registry(tmp_path_factory, monkeypatch):
    reg_dir = tmp_path_factory.mktemp("codesearch_reg")
    monkeypatch.setenv("CODESEARCH_REGISTRY", str(reg_dir / "indexes.json"))
    monkeypatch.delenv("CODESEARCH_INDEXES", raising=False)
    monkeypatch.delenv("CODESEARCH_MCP_LOG_DIR", raising=False)
    yield
    from codesearch import mcp_server

    mcp_server.clear_caches()
