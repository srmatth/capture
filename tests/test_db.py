"""DB layer smoke tests. Fast, no external services."""

from __future__ import annotations

from pathlib import Path

import pytest


def _import_db():
    from app import db  # noqa: WPS433 — deliberate late import after CONFIG rebind
    return db


def test_init_db_is_idempotent(tmp_data_root: Path) -> None:
    db = _import_db()
    db.init_db()
    db.init_db()  # second call must be a no-op, not a duplicate-migration error
    # And the schema tables exist:
    import sqlite3
    conn = sqlite3.connect(tmp_data_root / "library.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"items", "item_tags", "item_entities", "moves"} <= tables


def test_insert_and_get_item(tmp_data_root: Path) -> None:
    db = _import_db()
    db.init_db()
    db.insert_item(
        item_id="01TESTITEM000000000000000A",
        source_kind="image",
        original_filename="scan.jpg",
        mime_type="image/jpeg",
        size_bytes=1234,
        upload_note="journal from today",
    )
    row = db.get_item("01TESTITEM000000000000000A")
    assert row is not None
    assert row["source_kind"] == "image"
    assert row["status"] == "queued"
    assert row["upload_note"] == "journal from today"
    assert row["size_bytes"] == 1234


def test_update_item_whitelist(tmp_data_root: Path) -> None:
    db = _import_db()
    db.init_db()
    db.insert_item(
        item_id="01TESTITEM000000000000000B",
        source_kind="audio", original_filename="memo.m4a",
        mime_type="audio/m4a", size_bytes=1,
    )
    db.update_item("01TESTITEM000000000000000B", status="transcribed")
    assert db.get_item("01TESTITEM000000000000000B")["status"] == "transcribed"

    # Attempting to write a non-whitelisted field must raise.
    with pytest.raises(ValueError):
        db.update_item("01TESTITEM000000000000000B", secret_field="oops")


def test_list_by_status(tmp_data_root: Path) -> None:
    db = _import_db()
    db.init_db()
    for i in range(3):
        db.insert_item(
            item_id=f"01ITEM0000000000000000000{i}",
            source_kind="image",
            original_filename=f"{i}.jpg",
            mime_type="image/jpeg",
            size_bytes=1,
        )
    db.update_item("01ITEM00000000000000000000", status="transcribed")
    queued = db.list_items_by_status("queued")
    assert {r["id"] for r in queued} == {
        "01ITEM00000000000000000001", "01ITEM00000000000000000002",
    }
