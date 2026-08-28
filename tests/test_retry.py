"""Retry policy + POST /jobs/<id>/retry endpoint tests.

Fast — no external services, everything runs on the in-memory FastAPI
test client + tmp DB.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ---------- retry policy unit tests ----------


def test_backoff_schedule_matches_spec(tmp_data_root) -> None:
    from app.retry import next_retry_delay

    # 60s * 2^retry_count, capped at 86400 (24h).
    assert next_retry_delay(0) == 60
    assert next_retry_delay(1) == 120
    assert next_retry_delay(2) == 240
    assert next_retry_delay(3) == 480
    assert next_retry_delay(4) == 960
    assert next_retry_delay(5) == 1920
    # Cap kicks in around retry 11 (60 * 2^11 = 122880 > 86400).
    assert next_retry_delay(11) == 86400
    assert next_retry_delay(100) == 86400


def test_is_ready_for_retry_time_gate(tmp_data_root) -> None:
    from app.retry import is_ready_for_retry

    now = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)

    # Never failed → immediately ready.
    assert is_ready_for_retry(0, None, now=now) is True

    # 30 seconds ago after 0 retries (wait=60) → not yet.
    thirty_sec_ago = (now - timedelta(seconds=30)).isoformat()
    assert is_ready_for_retry(0, thirty_sec_ago, now=now) is False

    # 61 seconds ago after 0 retries → ready.
    sixty_one_sec_ago = (now - timedelta(seconds=61)).isoformat()
    assert is_ready_for_retry(0, sixty_one_sec_ago, now=now) is True

    # After 3 retries, wait=480. 4 min ago → not yet.
    four_min_ago = (now - timedelta(minutes=4)).isoformat()
    assert is_ready_for_retry(3, four_min_ago, now=now) is False


def test_is_ready_for_retry_max_attempts_blocks(tmp_data_root) -> None:
    from app.retry import MAX_ATTEMPTS, is_ready_for_retry

    now = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
    year_ago = (now - timedelta(days=365)).isoformat()

    # Waited a year, but retry_count is at cap → still no retry.
    assert is_ready_for_retry(MAX_ATTEMPTS, year_ago, now=now) is False


def test_should_dead_letter_at_threshold(tmp_data_root) -> None:
    from app.retry import MAX_ATTEMPTS, should_dead_letter

    assert should_dead_letter(MAX_ATTEMPTS - 1) is False
    assert should_dead_letter(MAX_ATTEMPTS) is True
    assert should_dead_letter(MAX_ATTEMPTS + 1) is True


# ---------- record_worker_failure ----------


def test_record_worker_failure_bumps_count(tmp_data_root) -> None:
    from app.db import get_item, init_db, insert_item, record_worker_failure

    init_db()
    insert_item(item_id="01FAIL0000000000000000000A",
                source_kind="audio", original_filename="x.m4a",
                mime_type="audio/m4a", size_bytes=1)

    n, dead = record_worker_failure("01FAIL0000000000000000000A", "boom")
    assert (n, dead) == (1, False)
    row = get_item("01FAIL0000000000000000000A")
    assert row["retry_count"] == 1
    assert row["status"] == "failed"
    assert row["error_message"] == "boom"
    assert row["last_error_at"] is not None


def test_record_worker_failure_transitions_to_dead_letter(tmp_data_root) -> None:
    from app.db import get_item, init_db, insert_item, record_worker_failure
    from app.retry import MAX_ATTEMPTS

    init_db()
    insert_item(item_id="01DEAD0000000000000000000A",
                source_kind="audio", original_filename="x.m4a",
                mime_type="audio/m4a", size_bytes=1)

    for _ in range(MAX_ATTEMPTS - 1):
        record_worker_failure("01DEAD0000000000000000000A", "err")
    # One more brings us to the threshold → dead_letter.
    _, dead = record_worker_failure("01DEAD0000000000000000000A", "final")
    assert dead is True
    row = get_item("01DEAD0000000000000000000A")
    assert row["status"] == "dead_letter"
    assert row["retry_count"] == MAX_ATTEMPTS


def test_reset_retry_state_rewinds(tmp_data_root) -> None:
    from app.db import (
        get_item, init_db, insert_item, record_worker_failure, reset_retry_state,
    )

    init_db()
    insert_item(item_id="01RESET000000000000000000A",
                source_kind="image", original_filename="x.jpg",
                mime_type="image/jpeg", size_bytes=1)
    record_worker_failure("01RESET000000000000000000A", "boom")
    record_worker_failure("01RESET000000000000000000A", "boom again")
    reset_retry_state("01RESET000000000000000000A", "transcribed")

    row = get_item("01RESET000000000000000000A")
    assert row["status"] == "transcribed"
    assert row["retry_count"] == 0
    assert row["last_error_at"] is None
    assert row["error_message"] is None


# ---------- POST /jobs/<id>/retry ----------


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_retry_endpoint_rewinds_to_correct_stage(tmp_data_root: Path) -> None:
    """A failed row whose classify never succeeded (transcript_path set,
    path not set) should rewind to 'transcribed' so classify picks it up."""
    from app.db import (
        init_db, insert_item, record_worker_failure, update_item,
    )
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    insert_item(item_id=item_id, source_kind="image",
                original_filename="x.jpg", mime_type="image/jpeg", size_bytes=1)
    update_item(item_id, status="transcribed",
                transcript_path=f"processed/image/{item_id}.txt",
                transcript_source="tesseract")
    record_worker_failure(item_id, "classify boom")

    r = _client().post(f"/jobs/{item_id}/retry")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "transcribed"
    assert body["retry_from_stage"] == "transcribed"

    from app.db import get_item
    row = get_item(item_id)
    assert row["status"] == "transcribed"
    assert row["retry_count"] == 0

    # Marker recreated for classify.
    from app.config import CONFIG
    assert (CONFIG.data_root / "queue" / "classify" / item_id).exists()


def test_retry_endpoint_rewinds_embed_stage(tmp_data_root: Path) -> None:
    """A failed row with path set but not yet embedded rewinds to
    'classified' so embed picks it up."""
    from app.db import (
        init_db, insert_item, record_worker_failure, update_item,
    )
    from fastapi.testclient import TestClient
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    insert_item(item_id=item_id, source_kind="image",
                original_filename="x.jpg", mime_type="image/jpeg", size_bytes=1)
    update_item(item_id, status="classified",
                transcript_path=f"processed/notes/personal/{item_id}.txt",
                path="notes/personal", title="x", one_line_summary="x",
                confidence=0.9, classifier_version="v1")
    record_worker_failure(item_id, "embed boom")

    r = _client().post(f"/jobs/{item_id}/retry")
    body = r.json()
    assert body["status"] == "classified"
    assert body["retry_from_stage"] == "classified"

    from app.config import CONFIG
    assert (CONFIG.data_root / "queue" / "embed" / item_id).exists()


def test_retry_endpoint_recovers_dead_letter(tmp_data_root: Path) -> None:
    """Even dead_letter items can be manually retried — that's the whole
    point of a manual override."""
    from app.db import (
        get_item, init_db, insert_item, record_worker_failure, update_item,
    )
    from app.retry import MAX_ATTEMPTS
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    insert_item(item_id=item_id, source_kind="image",
                original_filename="x.jpg", mime_type="image/jpeg", size_bytes=1)
    update_item(item_id, transcript_path=f"processed/image/{item_id}.txt")
    for _ in range(MAX_ATTEMPTS):
        record_worker_failure(item_id, "err")
    assert get_item(item_id)["status"] == "dead_letter"

    r = _client().post(f"/jobs/{item_id}/retry")
    assert r.status_code == 200

    row = get_item(item_id)
    assert row["status"] == "transcribed"
    assert row["retry_count"] == 0


def test_retry_endpoint_rejects_still_processing(tmp_data_root: Path) -> None:
    """You can't retry an item that's actively in-flight or already
    successful — that would clobber real work."""
    from app.db import init_db, insert_item, update_item
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    insert_item(item_id=item_id, source_kind="image",
                original_filename="x.jpg", mime_type="image/jpeg", size_bytes=1)
    update_item(item_id, status="embedded")

    r = _client().post(f"/jobs/{item_id}/retry")
    assert r.status_code == 409


def test_retry_endpoint_404_unknown_item(tmp_data_root: Path) -> None:
    r = _client().post("/jobs/does-not-exist/retry")
    assert r.status_code == 404


# ---------- worker picks up eligible failed items on its own ----------


def test_transcribe_worker_picks_up_eligible_failed_item(
    tmp_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the exponential-backoff window elapses, the worker should
    process a 'failed' row without any manual intervention."""
    from app.db import (
        get_item, init_db, insert_item, record_worker_failure, update_item,
    )
    from app.workers import transcribe
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    src = tmp_data_root / "inbox" / "audio" / f"{item_id}.m4a"
    src.write_bytes(b"fake audio")
    insert_item(item_id=item_id, source_kind="audio",
                original_filename="src.m4a", mime_type="audio/m4a", size_bytes=1)
    # Simulate a previous failure.
    record_worker_failure(item_id, "transient network hiccup")

    # Force the last_error_at back 2 minutes so the backoff (60s at
    # retry_count=1 is 120s) is just past its threshold.
    two_min_ago = (datetime.now(timezone.utc) - timedelta(seconds=125)).isoformat()
    update_item(item_id, last_error_at=two_min_ago)

    # Stub transcribe_audio + the shell-out helpers so the worker runs
    # to success.
    monkeypatch.setattr(transcribe, "transcribe_audio",
                         lambda p: "recovered transcript")

    transcribe.main()

    row = get_item(item_id)
    assert row["status"] == "transcribed"
    # retry_count from the earlier failure stays — it's a historical
    # record, not a counter we reset on success. Manual /retry is the
    # only path that resets it. (This matches the spec docstring in
    # app/retry.py.)
    assert row["retry_count"] == 1
    assert row["transcript_source"] == "whisper.cpp"


def test_worker_does_not_pick_up_dead_letter(
    tmp_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dead-lettered items must not be automatically retried, no matter
    how much time has passed."""
    from app.db import (
        get_item, init_db, insert_item, record_worker_failure, update_item,
    )
    from app.retry import MAX_ATTEMPTS
    from app.workers import transcribe
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    src = tmp_data_root / "inbox" / "audio" / f"{item_id}.m4a"
    src.write_bytes(b"fake audio")
    insert_item(item_id=item_id, source_kind="audio",
                original_filename="src.m4a", mime_type="audio/m4a", size_bytes=1)
    for _ in range(MAX_ATTEMPTS):
        record_worker_failure(item_id, "boom")
    assert get_item(item_id)["status"] == "dead_letter"

    # Time-travel: force last_error_at way back.
    year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    update_item(item_id, last_error_at=year_ago)

    monkeypatch.setattr(transcribe, "transcribe_audio",
                         lambda p: "would-be transcript")

    transcribe.main()

    row = get_item(item_id)
    assert row["status"] == "dead_letter"      # unchanged
    assert row["transcript_path"] is None      # worker didn't touch it
