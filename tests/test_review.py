"""Inbox / review page + HTMX action fragments + weekly digest task."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _client():
    from app.main import app
    return TestClient(app)


def _insert(item_id: str, *, status: str, path: str | None = None,
             confidence: float | None = None,
             title: str = "test", summary: str = "",
             updated_at: str | None = None) -> None:
    """Direct DB insert bypassing the pipeline. Lets us pose the DB in
    any state we want for testing the review page."""
    from app.db import _connect, init_db
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO items (id, uploaded_at, source_kind, mime_type, "
            "size_bytes, status, updated_at, path, title, "
            "one_line_summary, confidence, classifier_version, "
            "transcript_source) "
            "VALUES (?, ?, 'image', 'image/jpeg', 1, ?, ?, ?, ?, ?, ?, "
            "'v1', 'tesseract')",
            (item_id, now, status, updated_at or now, path, title, summary,
             confidence),
        )


# ---------- inbox page rendering ----------


def test_inbox_page_empty(tmp_data_root: Path) -> None:
    r = _client().get("/inbox")
    assert r.status_code == 200
    assert "Nothing to review" in r.text


def test_inbox_page_shows_needs_review_bucket(tmp_data_root: Path) -> None:
    _insert("01RV0000000000000000000A", status="embedded", path="inbox",
             confidence=0.3, title="Uncertain thing")
    r = _client().get("/inbox")
    assert r.status_code == 200
    assert "Needs review" in r.text
    assert "Uncertain thing" in r.text


def test_inbox_page_shows_spot_check_bucket(tmp_data_root: Path) -> None:
    _insert("01RV0000000000000000000B", status="embedded",
             path="notes/personal", confidence=0.65, title="Low confidence file")
    r = _client().get("/inbox")
    assert "Spot check" in r.text
    assert "Low confidence file" in r.text


def test_inbox_page_shows_failed_bucket(tmp_data_root: Path) -> None:
    from app.db import _connect
    _insert("01RV0000000000000000000C", status="failed", title="Broken item")
    with _connect() as conn:
        conn.execute(
            "UPDATE items SET error_message = 'boom', retry_count = 2 "
            "WHERE id = ?", ("01RV0000000000000000000C",),
        )
    r = _client().get("/inbox")
    assert "Failed / stuck" in r.text
    assert "Broken item" in r.text
    assert "boom" in r.text


def test_inbox_page_excludes_deleted(tmp_data_root: Path) -> None:
    """Soft-deleted rows must not appear in any bucket."""
    from app.db import _connect
    _insert("01RV0000000000000000000D", status="embedded", path="inbox",
             title="Ghost item")
    with _connect() as conn:
        conn.execute("UPDATE items SET deleted_at = ? WHERE id = ?",
                     ("2026-08-28T00:00:00+00:00", "01RV0000000000000000000D"))
    r = _client().get("/inbox")
    assert "Ghost item" not in r.text


def test_inbox_page_excludes_high_confidence_files(tmp_data_root: Path) -> None:
    """A confidently-filed item should NOT appear in the spot-check bucket."""
    _insert("01RV0000000000000000000E", status="embedded",
             path="notes/personal", confidence=0.95, title="High confidence")
    r = _client().get("/inbox")
    assert "High confidence" not in r.text


# ---------- HTMX fragment endpoints ----------


def test_inbox_delete_returns_fragment(tmp_data_root: Path) -> None:
    from app.db import get_item
    _insert("01RV0000000000000000000F", status="embedded", path="inbox",
             title="Delete me")
    r = _client().post("/inbox/01RV0000000000000000000F/delete")
    assert r.status_code == 200
    assert "deleted" in r.text
    assert "review-card resolved" in r.text
    assert get_item("01RV0000000000000000000F")["deleted_at"] is not None


def test_inbox_retry_rewinds_stage(tmp_data_root: Path) -> None:
    from app.db import _connect, get_item

    _insert("01RV0000000000000000000G", status="failed", title="Retry me")
    with _connect() as conn:
        # A failed item needs at least transcript_path to rewind to
        # 'transcribed'; otherwise it rewinds to 'queued'.
        conn.execute(
            "UPDATE items SET transcript_path = ? WHERE id = ?",
            ("processed/image/01RV0000000000000000000G.txt",
             "01RV0000000000000000000G"),
        )

    r = _client().post("/inbox/01RV0000000000000000000G/retry")
    assert r.status_code == 200
    assert "retry queued" in r.text

    row = get_item("01RV0000000000000000000G")
    assert row["status"] == "transcribed"
    assert row["retry_count"] == 0


def test_inbox_move_delegates_correctly(tmp_data_root: Path) -> None:
    """Move via HTMX endpoint should behave like the direct move — files
    relocate, DB updates, moves-audit records reason='user'."""
    from app.config import CONFIG
    from app.db import init_db, insert_item, set_tags, update_item, upsert_fts

    init_db()
    item_id = "01RV0000000000000000000H"
    insert_item(item_id=item_id, source_kind="image", original_filename="x.jpg",
                mime_type="image/jpeg", size_bytes=1)

    # Set up as classified in inbox/ (like the classify worker leaves
    # low-confidence items).
    (CONFIG.data_root / "inbox").mkdir(parents=True, exist_ok=True)
    (CONFIG.data_root / "inbox" / f"{item_id}.jpg").write_bytes(b"raw")
    (CONFIG.data_root / "processed" / "inbox").mkdir(parents=True, exist_ok=True)
    (CONFIG.data_root / "processed" / "inbox" / f"{item_id}.txt").write_text("t")
    update_item(item_id, status="embedded", path="inbox",
                final_filename=f"{item_id}.jpg", title="Move me",
                one_line_summary="", confidence=0.4,
                transcript_path=f"processed/inbox/{item_id}.txt")

    r = _client().post(f"/inbox/{item_id}/move",
                        data={"path": "notes/professional"})
    assert r.status_code == 200
    assert "moved to notes/professional" in r.text

    # File actually relocated.
    assert (CONFIG.data_root / "notes/professional" / f"{item_id}.jpg").exists()


def test_inbox_accept_records_audit_no_op(tmp_data_root: Path) -> None:
    """Accept just writes to the moves audit; nothing physically moves."""
    from app.db import _connect
    _insert("01RV0000000000000000000I", status="embedded",
             path="notes/personal", confidence=0.7, title="Accept me")

    r = _client().post("/inbox/01RV0000000000000000000I/accept")
    assert r.status_code == 200
    assert "accepted" in r.text

    with _connect() as conn:
        moves = conn.execute(
            "SELECT reason FROM moves WHERE item_id = ?",
            ("01RV0000000000000000000I",),
        ).fetchall()
    assert len(moves) == 1
    assert moves[0]["reason"] == "user_accept"


# ---------- weekly digest ----------


def test_weekly_digest_composition(tmp_data_root: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """compose_digest returns real numbers matching the seeded state."""
    _insert("01WK0000000000000000000A", status="embedded",
             path="notes/personal", confidence=0.9, title="Filed A")
    _insert("01WK0000000000000000000B", status="embedded",
             path="notes/professional", confidence=0.85, title="Filed B")
    _insert("01WK0000000000000000000C", status="embedded", path="inbox",
             confidence=0.4, title="Uncertain")
    _insert("01WK0000000000000000000D", status="embedded",
             path="reference/legal", confidence=0.7, title="Low conf")

    # Weekly digest reads from the running env — set what it wants.
    monkeypatch.setenv("NTFY_PASSWORD", "not-used-in-composition")

    from app.tasks.weekly_digest import compose_digest
    body, stats = compose_digest()

    assert stats["total_filed"] == 4
    assert stats["per_category"] == {
        "notes": 2, "inbox": 1, "reference": 1,
    }
    assert stats["inbox_pending"] == 1
    assert stats["spot_check"] == 1  # Only the confidence 0.7 one
    assert "Filed:" in body
    assert "notes" in body


def test_weekly_digest_zero_activity(tmp_data_root: Path) -> None:
    """An empty week still produces a useful digest."""
    from app.db import init_db
    init_db()
    from app.tasks.weekly_digest import compose_digest
    body, stats = compose_digest()
    assert stats["total_filed"] == 0
    assert "0 items this week" in body
    assert "Corpus: 0 items" in body


def test_weekly_digest_pushes_to_ntfy(tmp_data_root: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """main() calls push_ntfy exactly once with the composed body."""
    from app.db import init_db
    init_db()

    from app.tasks import weekly_digest
    calls = {}

    def fake_push(body: str) -> None:
        calls["body"] = body
        calls["count"] = calls.get("count", 0) + 1

    monkeypatch.setattr(weekly_digest, "push_ntfy", fake_push)
    rc = weekly_digest.main()

    assert rc == 0
    assert calls["count"] == 1
    assert "Week of" in calls["body"]
