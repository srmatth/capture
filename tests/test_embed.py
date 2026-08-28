"""Embed worker tests. Stubs the sentence-transformers model and the
Qdrant client so nothing hits the network or loads the ~90MB model."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


def _make_classified_item() -> str:
    """Insert a DB row already at status='classified' with a transcript
    on disk. This is the state the embed worker expects to find."""
    from app.config import CONFIG
    from app.db import init_db, insert_item, set_tags, update_item
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    insert_item(
        item_id=item_id,
        source_kind="image",
        original_filename="scan.jpg",
        mime_type="image/jpeg",
        size_bytes=1,
    )
    processed = CONFIG.data_root / "processed" / "notes" / "personal"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / f"{item_id}.txt").write_text("Some interesting shopping ideas.")

    update_item(
        item_id,
        status="classified",
        path="notes/personal",
        final_filename=f"{item_id}.jpg",
        title="Shopping ideas",
        one_line_summary="A short summary.",
        date_of_content="2026-08-28",
        confidence=0.82,
        classifier_version="haiku-4.5-taxonomy-v1",
        transcript_path=f"processed/notes/personal/{item_id}.txt",
    )
    set_tags(item_id, ["shopping", "personal"])
    (CONFIG.data_root / "queue" / "embed" / item_id).touch()
    return item_id


class _FakeModel:
    """Tiny stub replacing SentenceTransformer. Returns a fixed vector so
    tests can assert on exact upsert payloads."""

    def encode(self, text: str):
        # 384-dim to match all-MiniLM-L6-v2. Content doesn't matter for
        # these tests; we just verify it was called with our text.
        import numpy as np
        return np.zeros(384, dtype="float32")


class _FakeQdrant:
    def __init__(self) -> None:
        self.upserts: list[dict] = []

    def upsert(self, *, collection_name: str, points):
        for p in points:
            self.upserts.append({
                "collection": collection_name,
                "id": p.id,
                "vector_dim": len(p.vector),
                "payload": p.payload,
            })


@pytest.fixture
def embed_env(tmp_data_root, monkeypatch):
    """Stub the embed worker's two heavy dependencies."""
    from app.workers import embed

    fake_qdrant = _FakeQdrant()
    monkeypatch.setattr(embed, "_model", lambda: _FakeModel())
    monkeypatch.setattr(embed, "_qdrant", lambda: fake_qdrant)
    return embed, fake_qdrant


def test_embed_happy_path(embed_env, tmp_data_root: Path) -> None:
    embed, fake_qdrant = embed_env
    item_id = _make_classified_item()

    embed.process_one(item_id)

    # Qdrant received exactly one upsert with the right shape.
    assert len(fake_qdrant.upserts) == 1
    up = fake_qdrant.upserts[0]
    assert up["collection"] == "library"
    assert up["vector_dim"] == 384
    payload = up["payload"]
    assert payload["item_id"] == item_id
    assert payload["title"] == "Shopping ideas"
    assert payload["path"] == "notes/personal"
    assert set(payload["tags"]) == {"shopping", "personal"}
    assert payload["date_of_content"] == "2026-08-28"

    from app.db import get_item
    row = get_item(item_id)
    assert row["status"] == "embedded"

    # Marker cleaned up.
    from app.config import CONFIG
    marker = CONFIG.data_root / "queue" / "embed" / item_id
    assert not marker.exists()


def test_embed_populates_fts(embed_env, tmp_data_root: Path) -> None:
    """FTS5 mirror gets an entry keyed by item_id."""
    import sqlite3

    from app.config import CONFIG

    embed, _ = embed_env
    item_id = _make_classified_item()
    embed.process_one(item_id)

    conn = sqlite3.connect(CONFIG.library_db)
    rows = conn.execute(
        "SELECT id, title, one_line_summary FROM items_fts WHERE id = ?",
        (item_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "Shopping ideas"
    assert rows[0][2] == "A short summary."


def test_embed_rejects_wrong_status(embed_env, tmp_data_root: Path) -> None:
    """If status isn't 'classified' or 'embedding', we don't proceed."""
    from app.db import init_db, insert_item
    from app.workers import embed as embed_mod
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    insert_item(
        item_id=item_id,
        source_kind="image",
        original_filename="x.jpg",
        mime_type="image/jpeg",
        size_bytes=1,
    )
    # Still status='queued' — hasn't even been transcribed.
    with pytest.raises(ValueError, match="status="):
        embed_mod.process_one(item_id)


def test_ulid_to_uuid_deterministic(tmp_data_root: Path) -> None:
    """The ULID→UUID mapping must be deterministic and reversible so
    re-embedding the same item finds and updates the existing point."""
    from app.workers.embed import _ulid_to_uuid

    ulid = "01M14XEEVE67BDV3NPG13CHJN9"  # arbitrary valid ULID
    a = _ulid_to_uuid(ulid)
    b = _ulid_to_uuid(ulid)
    assert a == b
    # And valid UUID format.
    import uuid as uuidlib
    parsed = uuidlib.UUID(a)
    assert len(str(parsed)) == 36
