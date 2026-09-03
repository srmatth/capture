"""Tests for the book excerpt linking integration."""
from __future__ import annotations

import io

from fastapi.testclient import TestClient


def _init():
    from app.db import init_db
    from app.taxonomy import seed_builtins
    init_db()
    seed_builtins()


def _client():
    from app.main import app
    return TestClient(app)


def test_save_book_link(tmp_data_root):
    _init()
    from app.db import get_item, insert_item, save_book_link

    insert_item(
        item_id="BOOK01",
        source_kind="image",
        original_filename="page.jpg",
        mime_type="image/jpeg",
        size_bytes=1000,
    )
    save_book_link("BOOK01", book_id=42, reading_id=57, book_title="The Black Swan")

    item = get_item("BOOK01")
    assert item["reading_book_id"] == 42
    assert item["reading_reading_id"] == 57
    assert item["reading_book_title"] == "The Black Swan"


def test_upload_with_book_shortcut(tmp_data_root):
    from app.db import get_item

    client = _client()
    resp = client.post(
        "/upload?book_id=10&reading_id=20&book_title=Test+Book",
        files={"file": ("test.jpg", io.BytesIO(b"fake image data"), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    item = get_item(data["id"])
    assert item["reading_book_id"] == 10
    assert item["reading_reading_id"] == 20
    assert item["reading_book_title"] == "Test Book"


def test_upload_without_book_shortcut(tmp_data_root):
    from app.db import get_item

    client = _client()
    resp = client.post(
        "/upload",
        files={"file": ("test.jpg", io.BytesIO(b"fake image data"), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    item = get_item(data["id"])
    assert item["reading_book_id"] is None


def test_media_books_in_taxonomy(tmp_data_root):
    _init()
    from app.taxonomy import get_taxonomy
    taxonomy = get_taxonomy()
    assert "media/books" in taxonomy


def test_build_prompt_with_extra_context(tmp_data_root):
    _init()
    from app.taxonomy import build_classify_prompt
    prompt = build_classify_prompt("some item text", extra_context="Currently reading: Test Book")
    assert "Currently reading: Test Book" in prompt
    assert "some item text" in prompt


def test_build_prompt_without_extra_context(tmp_data_root):
    _init()
    from app.taxonomy import build_classify_prompt
    prompt = build_classify_prompt("some item text")
    assert "some item text" in prompt
    assert "book_id" in prompt
