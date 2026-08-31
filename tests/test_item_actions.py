"""Item detail endpoints: raw serve, move, delete/undelete, reclassify."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _client():
    """TestClient with a JSON accept header by default. The action
    endpoints (delete, move, reclassify) 303-redirect browser POSTs
    but return JSON when the caller specifies Accept: application/json.
    Tests are programmatic callers, so we want the JSON path."""
    from app.main import app
    return TestClient(app, headers={"accept": "application/json"})


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


# ---------- redirect behavior for browser POSTs ----------
#
# Browser form submissions (Accept: text/html) that hit the item-action
# endpoints must land somewhere useful, not on a raw JSON blob. Previously
# clicking Delete on a phone dumped you on {"deleted": true} and required
# a force-quit to recover.


def _browser_client():
    """TestClient with a browser-like Accept header (no application/json)."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app, headers={"accept": "text/html,*/*"})


def test_delete_redirects_browser_to_browse_path(tmp_data_root: Path) -> None:
    item_id = "01RED0000000000000000000A"
    _seed_classified_item(item_id=item_id, path="notes/personal")

    r = _browser_client().post(f"/item/{item_id}/delete", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/browse?path=notes/personal"


def test_move_redirects_browser_to_item_detail(tmp_data_root: Path) -> None:
    item_id = "01RED0000000000000000000B"
    _seed_classified_item(item_id=item_id, path="notes/personal")

    r = _browser_client().post(
        f"/item/{item_id}/move", data={"path": "notes/professional"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/item/{item_id}"


def test_reclassify_redirects_browser(tmp_data_root: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from app.workers import classify
    monkeypatch.setattr(classify, "process_one", lambda _i: None)

    item_id = "01RED0000000000000000000C"
    _seed_classified_item(item_id=item_id)

    r = _browser_client().post(
        f"/item/{item_id}/reclassify", follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/item/{item_id}"


def test_undelete_redirects_browser(tmp_data_root: Path) -> None:
    item_id = "01RED0000000000000000000D"
    _seed_classified_item(item_id=item_id)
    _client().post(f"/item/{item_id}/delete")   # soft-delete via JSON client

    r = _browser_client().post(
        f"/item/{item_id}/undelete", follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/item/{item_id}"


def test_json_client_still_gets_json(tmp_data_root: Path) -> None:
    """Programmatic callers (like the inbox HTMX endpoints and tests)
    that set Accept: application/json still get the JSON payload back."""
    item_id = "01RED0000000000000000000E"
    _seed_classified_item(item_id=item_id)

    r = _client().post(f"/item/{item_id}/delete")
    assert r.status_code == 200
    assert r.json() == {"deleted": True}


# ---------- retranscribe ----------


def test_retranscribe_with_vision_queues_hint(tmp_data_root: Path) -> None:
    """POST /item/<id>/retranscribe?with=vision should return immediately
    with a job handle after setting retranscribe_hint. The transcribe
    worker (not the endpoint) does the actual work asynchronously."""
    item_id = "01RTV0000000000000000000A"
    _seed_classified_item(item_id=item_id)  # source_kind='image', tesseract

    r = _client().post(f"/item/{item_id}/retranscribe?with=vision")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queued"] is True
    assert body["with"] == "vision"
    assert body["status_url"] == f"/jobs/{item_id}"

    from app.db import get_item
    row = get_item(item_id)
    assert row["retranscribe_hint"] == "vision"
    # Status rewound so the worker picks it up.
    assert row["status"] == "queued"


def test_retranscribe_worker_processes_hint(tmp_data_root: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """The transcribe worker picks up items with retranscribe_hint set,
    dispatches to the right method, and clears the hint on success."""
    from app.config import CONFIG
    from app.db import get_item
    from app.workers import transcribe

    item_id = "01RTW0000000000000000000A"
    _seed_classified_item(item_id=item_id)  # image, path='notes/personal'

    # Queue a vision retranscribe.
    _client().post(f"/item/{item_id}/retranscribe?with=vision")

    calls: list[list[str]] = []
    monkeypatch.setattr(
        transcribe, "_claude_transcribe",
        lambda paths: (calls.append([p.name for p in paths]) or "REVISED") or "REVISED",
    )

    # Worker sweep. Should find the hinted item and process it.
    transcribe.main()

    from app.db import get_item as _get_item
    row = _get_item(item_id)
    assert row["transcript_source"] == "claude-vision"
    assert row["retranscribe_hint"] is None, "hint should be cleared on success"
    assert row["status"] == "transcribed"

    # Handoff marker written for the classify stage.
    assert (CONFIG.data_root / "queue" / "classify" / item_id).exists()

    # Claude was called with the raw image.
    assert calls == [[f"{item_id}.jpg"]]


def test_retranscribe_audio_rejected(tmp_data_root: Path) -> None:
    """Audio items don't have a vision fallback — Whisper is the only
    audio transcription path."""
    from app.db import init_db, insert_item, update_item

    init_db()
    item_id = "01RTV0000000000000000000B"
    insert_item(item_id=item_id, source_kind="audio",
                 original_filename="x.m4a", mime_type="audio/m4a", size_bytes=1)
    update_item(item_id, status="embedded", path="notes/personal",
                 title="clip", transcript_path=f"processed/audio/{item_id}.txt",
                 transcript_source="whisper.cpp")

    r = _client().post(f"/item/{item_id}/retranscribe?with=vision")
    assert r.status_code == 409
    assert "audio" in r.text.lower()


def test_retranscribe_with_force_ocr_queues_hint(tmp_data_root: Path) -> None:
    """POST /item/<id>/retranscribe?with=force-ocr should queue the hint,
    not run OCR synchronously."""
    from app.config import CONFIG
    from app.db import get_item, init_db, insert_item, update_item

    init_db()
    item_id = "01FOO0000000000000000000A"
    insert_item(item_id=item_id, source_kind="pdf",
                 original_filename="printed.pdf",
                 mime_type="application/pdf", size_bytes=1)
    (CONFIG.data_root / "media" / "articles").mkdir(parents=True, exist_ok=True)
    (CONFIG.data_root / "media" / "articles" / f"{item_id}.pdf").write_bytes(b"%PDF")
    update_item(item_id, status="embedded", path="media/articles",
                 final_filename=f"{item_id}.pdf", title="Printed article",
                 transcript_path=f"processed/media/articles/{item_id}.txt",
                 transcript_source="tesseract-pdf")

    r = _client().post(f"/item/{item_id}/retranscribe?with=force-ocr")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queued"] is True
    assert body["with"] == "force-ocr"

    row = get_item(item_id)
    assert row["retranscribe_hint"] == "force-ocr"
    assert row["status"] == "queued"


def test_retranscribe_force_ocr_rejected_for_images(tmp_data_root: Path) -> None:
    """force-ocr is a PDF-specific concept; asking for it on an image
    item should 409 rather than silently falling back to tesseract."""
    item_id = "01FOO0000000000000000000B"
    _seed_classified_item(item_id=item_id)  # image

    r = _client().post(f"/item/{item_id}/retranscribe?with=force-ocr")
    assert r.status_code == 409
    assert "pdf-only" in r.text.lower() or "pdf" in r.text.lower()


def test_retranscribe_pdf_without_batch_pages_rejects_vision(
    tmp_data_root: Path,
) -> None:
    """A genuine (non-batch) PDF has no source images we could send to
    vision. The endpoint should reject with a clear message rather than
    silently no-op."""
    from app.config import CONFIG
    from app.db import init_db, insert_item, update_item

    init_db()
    item_id = "01RTV0000000000000000000C"
    insert_item(item_id=item_id, source_kind="pdf",
                 original_filename="paper.pdf", mime_type="application/pdf",
                 size_bytes=1)
    (CONFIG.data_root / "reference" / "academic").mkdir(parents=True, exist_ok=True)
    (CONFIG.data_root / "reference" / "academic" / f"{item_id}.pdf").write_bytes(b"%PDF")
    update_item(item_id, status="embedded", path="reference/academic",
                 final_filename=f"{item_id}.pdf", title="Paper",
                 transcript_path=f"processed/reference/academic/{item_id}.txt",
                 transcript_source="tesseract-pdf")

    r = _client().post(f"/item/{item_id}/retranscribe?with=vision")
    assert r.status_code == 409
    assert "non-batch" in r.text.lower() or "vision" in r.text.lower()


# ---------- edit metadata ----------


def test_edit_metadata_updates_fields(tmp_data_root: Path) -> None:
    """POST /item/<id>/edit updates title, summary, date, tags."""
    from app.db import get_item, get_tags

    item_id = "01EDT0000000000000000000A"
    _seed_classified_item(item_id=item_id)

    r = _client().post(
        f"/item/{item_id}/edit",
        data={
            "title": "New title",
            "one_line_summary": "New one-liner.",
            "date_of_content": "2026-03-15",
            "tags": "antitrust, legal, brief",
        },
    )
    assert r.status_code == 200
    assert r.json()["edited"] is True

    row = get_item(item_id)
    assert row["title"] == "New title"
    assert row["one_line_summary"] == "New one-liner."
    assert row["date_of_content"] == "2026-03-15"
    assert set(get_tags(item_id)) == {"antitrust", "legal", "brief"}


def test_edit_metadata_only_touches_submitted_fields(tmp_data_root: Path) -> None:
    """A form that submits only title shouldn't nuke the summary."""
    from app.db import get_item

    item_id = "01EDT0000000000000000000B"
    _seed_classified_item(item_id=item_id)

    original = get_item(item_id)
    _client().post(f"/item/{item_id}/edit", data={"title": "Changed only title"})

    row = get_item(item_id)
    assert row["title"] == "Changed only title"
    assert row["one_line_summary"] == original["one_line_summary"]


def test_edit_metadata_rejects_bad_date(tmp_data_root: Path) -> None:
    item_id = "01EDT0000000000000000000C"
    _seed_classified_item(item_id=item_id)

    r = _client().post(
        f"/item/{item_id}/edit",
        data={"date_of_content": "not-a-date"},
    )
    assert r.status_code == 400


def test_edit_metadata_can_clear_optional_fields(tmp_data_root: Path) -> None:
    """Empty string on nullable fields clears them to NULL. Real HTML
    forms always transmit the field with an empty value; this test
    calls the endpoint directly at the Python level rather than going
    through the httpx2 TestClient, which strips empty form values.
    """
    from app.db import get_item, update_item
    from app.routers.search import item_edit

    item_id = "01EDT0000000000000000000D"
    _seed_classified_item(item_id=item_id)
    update_item(item_id, date_of_content="2026-01-01")

    # Fake Request with JSON accept so the response is plain dict.
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as _:
        pass  # ensure app is initialized
    from starlette.requests import Request as StarletteRequest
    req = StarletteRequest({
        "type": "http", "method": "POST", "headers": [(b"accept", b"application/json")],
    })

    import asyncio
    asyncio.run(item_edit(request=req, item_id=item_id, date_of_content=""))

    assert get_item(item_id)["date_of_content"] is None


def test_edit_metadata_refreshes_fts(tmp_data_root: Path) -> None:
    """After editing title/summary, the FTS index should reflect the new
    text so keyword search finds items by their edited fields."""
    from app.search import fts_search

    item_id = "01EDT0000000000000000000E"
    _seed_classified_item(item_id=item_id)

    _client().post(
        f"/item/{item_id}/edit",
        data={"title": "zumbleflorph antitrust"},
    )
    hits = fts_search("zumbleflorph")
    ids = [h[0] for h in hits]
    assert item_id in ids


# ---------- comments ----------


def test_add_comment_appears_in_get_comments(tmp_data_root: Path) -> None:
    from app.db import get_comments

    item_id = "01CMT0000000000000000000A"
    _seed_classified_item(item_id=item_id)

    r = _client().post(
        f"/item/{item_id}/comments",
        data={"body": "This is a note about the item."},
    )
    assert r.status_code == 200
    assert r.json()["added_comment"] is True

    comments = get_comments(item_id)
    assert len(comments) == 1
    assert comments[0]["body"] == "This is a note about the item."
    assert comments[0]["created_at"]


def test_comments_are_ordered(tmp_data_root: Path) -> None:
    from app.db import get_comments

    item_id = "01CMT0000000000000000000B"
    _seed_classified_item(item_id=item_id)

    for i in range(3):
        _client().post(f"/item/{item_id}/comments", data={"body": f"comment #{i}"})

    comments = get_comments(item_id)
    assert [c["body"] for c in comments] == ["comment #0", "comment #1", "comment #2"]


def test_delete_comment_removes_it(tmp_data_root: Path) -> None:
    from app.db import get_comments

    item_id = "01CMT0000000000000000000C"
    _seed_classified_item(item_id=item_id)
    _client().post(f"/item/{item_id}/comments", data={"body": "delete me"})

    comment_id = get_comments(item_id)[0]["id"]
    r = _client().post(f"/item/{item_id}/comments/{comment_id}/delete")
    assert r.status_code == 200

    assert get_comments(item_id) == []


def test_empty_comment_rejected(tmp_data_root: Path) -> None:
    item_id = "01CMT0000000000000000000D"
    _seed_classified_item(item_id=item_id)

    r = _client().post(f"/item/{item_id}/comments", data={"body": "   "})
    assert r.status_code == 400


def test_comments_cascade_on_hard_item_delete(tmp_data_root: Path) -> None:
    """FK ON DELETE CASCADE test. Soft-delete via /delete doesn't remove
    the DB row, but a direct DELETE cascades to item_comments."""
    from app.db import _connect, add_comment, get_comments

    item_id = "01CMT0000000000000000000E"
    _seed_classified_item(item_id=item_id)
    add_comment(item_id, "will be cascaded")
    assert len(get_comments(item_id)) == 1

    with _connect() as conn:
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
    assert get_comments(item_id) == []
