"""End-to-end page-render smoke tests. These aren't checking layout —
they're checking that every page returns 200 with a body that includes
expected content. This is the class of test that catches TemplateResponse
API breaks (the old {'request': ...} positional arg style is deprecated
and now raises), missing templates, and NameErrors in the Jinja context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _client():
    from app.main import app
    return TestClient(app)


def _seed_full_item(*, item_id: str, path: str = "notes/personal") -> None:
    """Populate one classified + embedded item so the pages have real
    content to render."""
    from app.config import CONFIG
    from app.db import init_db, insert_item, set_tags, update_item, upsert_fts

    init_db()
    insert_item(item_id=item_id, source_kind="image",
                original_filename="src.jpg", mime_type="image/jpeg",
                size_bytes=1)
    (CONFIG.data_root / path).mkdir(parents=True, exist_ok=True)
    (CONFIG.data_root / path / f"{item_id}.jpg").write_bytes(b"fake jpeg")
    (CONFIG.data_root / "processed" / path).mkdir(parents=True, exist_ok=True)
    (CONFIG.data_root / "processed" / path / f"{item_id}.txt").write_text("transcript text")

    update_item(item_id, status="embedded", path=path,
                final_filename=f"{item_id}.jpg", title="A test item",
                one_line_summary="A one-line summary.",
                date_of_content="2026-08-28", confidence=0.85,
                classifier_version="v1",
                transcript_path=f"processed/{path}/{item_id}.txt",
                transcript_char_count=15, transcript_source="tesseract")
    set_tags(item_id, ["personal", "test"])
    upsert_fts(item_id, title="A test item", summary="A one-line summary.",
                transcript="transcript text")


# ---------- landing pages ----------


def test_capture_root_serves_upload_pwa(tmp_data_root: Path) -> None:
    """GET / on the default host serves the upload PWA (index.html)."""
    r = _client().get("/")
    assert r.status_code == 200
    assert "capture-btn" in r.text or "Scan document" in r.text


def test_search_landing_empty(tmp_data_root: Path) -> None:
    r = _client().get("/search")
    assert r.status_code == 200
    assert "Search" in r.text


def test_search_landing_with_query(tmp_data_root: Path) -> None:
    _seed_full_item(item_id="01PAG0000000000000000000A")
    r = _client().get("/search?q=test item")
    assert r.status_code == 200
    assert "A test item" in r.text


def test_search_via_search_subdomain_host_header(tmp_data_root: Path) -> None:
    """The Host-header switch in main.py: / on search.matthewshome should
    serve the search page, not the upload PWA."""
    r = _client().get("/", headers={"host": "search.matthewshome"})
    assert r.status_code == 200
    assert "Search" in r.text or "search-form" in r.text


# ---------- browse ----------


def test_browse_root(tmp_data_root: Path) -> None:
    _seed_full_item(item_id="01PAG0000000000000000000B")
    r = _client().get("/browse")
    assert r.status_code == 200
    assert "notes" in r.text  # taxonomy facet


def test_browse_path(tmp_data_root: Path) -> None:
    _seed_full_item(item_id="01PAG0000000000000000000C", path="notes/personal")
    r = _client().get("/browse?path=notes/personal")
    assert r.status_code == 200
    assert "A test item" in r.text


# ---------- tag ----------


def test_tag_view(tmp_data_root: Path) -> None:
    _seed_full_item(item_id="01PAG0000000000000000000D")
    r = _client().get("/tags/personal")
    assert r.status_code == 200
    assert "A test item" in r.text


# ---------- item detail ----------


def test_item_detail(tmp_data_root: Path) -> None:
    item_id = "01PAG0000000000000000000E"
    _seed_full_item(item_id=item_id)
    r = _client().get(f"/item/{item_id}")
    assert r.status_code == 200
    assert "A test item" in r.text
    assert "transcript text" in r.text
    assert "personal" in r.text  # tag
    assert "notes/personal" in r.text  # path


def test_item_detail_404(tmp_data_root: Path) -> None:
    r = _client().get("/item/does-not-exist")
    assert r.status_code == 404
