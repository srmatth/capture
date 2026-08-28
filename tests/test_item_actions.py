"""Item detail endpoints: raw serve, move, delete/undelete, reclassify."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _client():
    from app.main import app
    return TestClient(app)


def _seed_classified_item(*, item_id: str, path: str = "notes/personal") -> None:
    """Set up an item as if it had made it through classify successfully:
    - Raw file at data/<path>/<id>.jpg
    - Transcript + meta.json at data/processed/<path>/
    - DB row status='classified'
    """
    from app.config import CONFIG
    from app.db import init_db, insert_item, set_tags, update_item, upsert_fts

    init_db()
    insert_item(item_id=item_id, source_kind="image",
                original_filename="src.jpg", mime_type="image/jpeg",
                size_bytes=1)

    (CONFIG.data_root / path).mkdir(parents=True, exist_ok=True)
    (CONFIG.data_root / path / f"{item_id}.jpg").write_bytes(b"fake jpeg")

    processed = CONFIG.data_root / "processed" / path
    processed.mkdir(parents=True, exist_ok=True)
    (processed / f"{item_id}.txt").write_text("transcript body")
    (processed / f"{item_id}.meta.json").write_text(json.dumps({
        "id": item_id, "path": path, "title": "X",
    }))

    update_item(item_id, status="classified", path=path,
                final_filename=f"{item_id}.jpg", title="X",
                one_line_summary="one-liner", confidence=0.85,
                classifier_version="v1",
                transcript_path=f"processed/{path}/{item_id}.txt",
                transcript_char_count=15, transcript_source="tesseract")
    set_tags(item_id, ["personal"])
    upsert_fts(item_id, title="X", summary="one-liner", transcript="transcript body")


# ---------- raw serve ----------


def test_item_raw_serves_file(tmp_data_root: Path) -> None:
    item_id = "01RAW0000000000000000000A"
    _seed_classified_item(item_id=item_id)

    r = _client().get(f"/item/{item_id}/raw")
    assert r.status_code == 200
    assert r.content == b"fake jpeg"


def test_item_raw_404_when_missing(tmp_data_root: Path) -> None:
    r = _client().get("/item/does-not-exist/raw")
    assert r.status_code == 404


# ---------- move ----------


def test_item_move_relocates_raw_and_transcript(tmp_data_root: Path) -> None:
    from app.config import CONFIG
    from app.db import get_item

    item_id = "01MOV0000000000000000000A"
    _seed_classified_item(item_id=item_id, path="notes/personal")

    r = _client().post(f"/item/{item_id}/move", data={"path": "notes/professional"})
    assert r.status_code == 200
    body = r.json()
    assert body["moved"] is True
    assert body["from"] == "notes/personal"
    assert body["to"] == "notes/professional"

    # Raw moved.
    assert (CONFIG.data_root / "notes/professional" / f"{item_id}.jpg").exists()
    assert not (CONFIG.data_root / "notes/personal" / f"{item_id}.jpg").exists()

    # Transcript + meta.json moved.
    new_txt = CONFIG.data_root / "processed/notes/professional" / f"{item_id}.txt"
    new_meta = CONFIG.data_root / "processed/notes/professional" / f"{item_id}.meta.json"
    assert new_txt.exists()
    assert new_meta.exists()

    # DB row updated.
    row = get_item(item_id)
    assert row["path"] == "notes/professional"

    # meta.json's `path` field was updated too.
    meta = json.loads(new_meta.read_text())
    assert meta["path"] == "notes/professional"


def test_item_move_records_audit_entry(tmp_data_root: Path) -> None:
    from app.db import _connect

    item_id = "01AUD0000000000000000000A"
    _seed_classified_item(item_id=item_id, path="notes/personal")

    _client().post(f"/item/{item_id}/move", data={"path": "reference/legal"})

    with _connect() as conn:
        moves = conn.execute(
            "SELECT from_path, to_path, reason FROM moves WHERE item_id = ?",
            (item_id,),
        ).fetchall()
    # There's one 'classify' auto-move from setup (no — setup calls
    # update_item directly, not record_move) plus the user move here.
    assert len(moves) == 1
    assert moves[0]["from_path"] == "notes/personal"
    assert moves[0]["to_path"] == "reference/legal"
    assert moves[0]["reason"] == "user"


def test_item_move_rejects_unknown_path(tmp_data_root: Path) -> None:
    item_id = "01BAD0000000000000000000A"
    _seed_classified_item(item_id=item_id)
    r = _client().post(f"/item/{item_id}/move", data={"path": "not-a-real-path"})
    assert r.status_code == 400


def test_item_move_same_path_is_noop(tmp_data_root: Path) -> None:
    item_id = "01SAM0000000000000000000A"
    _seed_classified_item(item_id=item_id, path="notes/personal")
    r = _client().post(f"/item/{item_id}/move", data={"path": "notes/personal"})
    assert r.status_code == 200
    assert r.json()["moved"] is False


# ---------- delete / undelete ----------


def test_item_delete_hides_from_search(tmp_data_root: Path) -> None:
    from app.db import get_item
    from app.search import list_by_tag

    item_id = "01DEL0000000000000000000A"
    _seed_classified_item(item_id=item_id)

    # Findable before.
    assert any(h.item_id == item_id for h in list_by_tag("personal"))

    r = _client().post(f"/item/{item_id}/delete")
    assert r.status_code == 200 and r.json()["deleted"] is True

    row = get_item(item_id)
    assert row["deleted_at"] is not None

    # No longer visible in search.
    assert list_by_tag("personal") == []


def test_item_undelete_restores(tmp_data_root: Path) -> None:
    from app.db import get_item
    from app.search import list_by_tag

    item_id = "01UND0000000000000000000A"
    _seed_classified_item(item_id=item_id)
    _client().post(f"/item/{item_id}/delete")
    _client().post(f"/item/{item_id}/undelete")

    row = get_item(item_id)
    assert row["deleted_at"] is None
    assert any(h.item_id == item_id for h in list_by_tag("personal"))


# ---------- reclassify ----------


def test_item_reclassify_calls_worker(tmp_data_root: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from app.workers import classify

    item_id = "01REC0000000000000000000A"
    _seed_classified_item(item_id=item_id)

    calls = []
    monkeypatch.setattr(classify, "process_one",
                         lambda i: calls.append(i))

    r = _client().post(f"/item/{item_id}/reclassify")
    assert r.status_code == 200
    assert calls == [item_id]


def test_item_reclassify_404(tmp_data_root: Path) -> None:
    r = _client().post("/item/does-not-exist/reclassify")
    assert r.status_code == 404
