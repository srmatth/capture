"""Classify worker tests. Stubs the Anthropic call so nothing hits
the network. Focus is on the deterministic post-LLM logic: JSON
parsing, confidence floor, path resolution (including journal date
partitioning + project-name expansion), file moves, DB updates, marker
handoff."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------- helpers ----------


@pytest.fixture(autouse=True)
def _seed_taxonomy(tmp_data_root):
    """The classify worker reads the taxonomy from SQLite. Every test in
    this module needs the DB initialised and the built-in taxonomy seeded
    before it can call _resolve_path / process_one."""
    from app.db import init_db
    from app.taxonomy import seed_builtins
    init_db()
    seed_builtins()


def _make_item_ready_for_classify(*, note: str = "", uploaded_at: str | None = None) -> str:
    """Insert a DB row, drop a raw file in inbox/image/, and write a
    transcript at processed/image/<id>.txt so classify can read it."""
    from app.config import CONFIG
    from app.db import init_db, insert_item, update_item
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    if uploaded_at is None:
        uploaded_at = datetime.now(timezone.utc).isoformat()
    insert_item(
        item_id=item_id,
        source_kind="image",
        original_filename="scan.jpg",
        mime_type="image/jpeg",
        size_bytes=1,
        upload_note=note,
    )
    # Simulate the transcribe stage's outputs.
    src = CONFIG.data_root / "inbox" / "image" / f"{item_id}.jpg"
    src.write_bytes(b"fake image bytes")

    transcript_dir = CONFIG.data_root / "processed" / "image"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{item_id}.txt").write_text("This is the item text.")

    update_item(
        item_id,
        status="transcribed",
        transcript_path=str((transcript_dir / f"{item_id}.txt")
                            .relative_to(CONFIG.data_root)),
        transcript_char_count=22,
        transcript_source="tesseract",
    )
    (CONFIG.data_root / "queue" / "classify" / item_id).touch()
    return item_id


def _stub_haiku(monkeypatch: pytest.MonkeyPatch, response: dict) -> list:
    """Replace _call_haiku with a stub that returns `response` and
    records how many times it was called. Returns the call log."""
    from app.workers import classify
    calls: list = []

    def fake(transcript: str) -> dict:
        calls.append(transcript)
        return response

    monkeypatch.setattr(classify, "_call_haiku", fake)
    return calls


# ---------- resolve_path unit tests (fast, no LLM) ----------


def test_resolve_path_low_confidence_forces_inbox(tmp_data_root) -> None:
    from app.workers.classify import _resolve_path

    assert _resolve_path("notes/personal", 0.5, "01ABC", "2026-08-28T00:00:00Z", None) == "inbox"
    assert _resolve_path("notes/personal", 0.6, "01ABC", "2026-08-28T00:00:00Z", None) == "notes/personal"


def test_resolve_path_unknown_path_goes_inbox(tmp_data_root) -> None:
    from app.workers.classify import _resolve_path

    # LLM invented a category we didn't offer.
    assert _resolve_path("hobbies/knitting", 0.99, "01A", "2026-08-28T00:00:00Z", None) == "inbox"


def test_resolve_path_journal_expands_by_date_of_content(tmp_data_root) -> None:
    from app.workers.classify import _resolve_path

    got = _resolve_path("journal", 0.9, "01A", "2026-08-28T00:00:00Z", "2026-03-15")
    assert got == "journal/2026/03"


def test_resolve_path_journal_falls_back_to_upload_date(tmp_data_root) -> None:
    from app.workers.classify import _resolve_path

    got = _resolve_path("journal", 0.9, "01A", "2026-08-28T00:00:00Z", None)
    assert got == "journal/2026/08"


def test_resolve_path_project_leaf_accepted(tmp_data_root) -> None:
    from app.workers.classify import _resolve_path

    got = _resolve_path("notes/project/tax-2026", 0.9, "01A", "2026-08-28T00:00:00Z", None)
    assert got == "notes/project/tax-2026"


def test_resolve_path_project_leaf_with_weird_chars_rejected(tmp_data_root) -> None:
    from app.workers.classify import _resolve_path

    # Anything that isn't alphanumeric-hyphen-underscore should NOT be
    # trusted as a project name.
    got = _resolve_path("notes/project/foo/bar", 0.9, "01A", "2026-08-28T00:00:00Z", None)
    assert got == "inbox"


# ---------- end-to-end process_one tests (LLM stubbed) ----------


def test_classify_notes_personal_happy_path(tmp_data_root: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import CONFIG
    from app.db import get_item, get_tags
    from app.workers import classify

    item_id = _make_item_ready_for_classify()

    calls = _stub_haiku(monkeypatch, {
        "path": "notes/personal",
        "title": "Shopping ideas",
        "one_line_summary": "Random shopping ideas.",
        "tags": ["shopping", "personal"],
        "date_of_content": None,
        "confidence": 0.82,
        "entities": {"person": ["Alice"]},
    })

    classify.process_one(item_id)

    assert len(calls) == 1

    row = get_item(item_id)
    assert row["status"] == "classified"
    assert row["path"] == "notes/personal"
    assert row["title"] == "Shopping ideas"
    assert row["one_line_summary"] == "Random shopping ideas."
    assert row["confidence"] == pytest.approx(0.82)
    assert row["classifier_version"]

    # Raw file was moved to <path>/<id>.<ext>.
    assert (CONFIG.data_root / "notes/personal" / f"{item_id}.jpg").exists()
    assert not (CONFIG.data_root / "inbox" / "image" / f"{item_id}.jpg").exists()

    # Meta.json + transcript relocated to processed/<path>/.
    assert (CONFIG.data_root / "processed" / "notes/personal" / f"{item_id}.txt").exists()
    meta_path = CONFIG.data_root / "processed" / "notes/personal" / f"{item_id}.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["path"] == "notes/personal"
    assert meta["tags"] == ["shopping", "personal"]

    # Tags stored in DB.
    assert set(get_tags(item_id)) == {"shopping", "personal"}

    # Handoff marker + no lingering classify marker.
    assert (CONFIG.data_root / "queue" / "embed" / item_id).exists()
    assert not (CONFIG.data_root / "queue" / "classify" / item_id).exists()


def test_classify_journal_partitioned_by_date_of_content(tmp_data_root: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import CONFIG
    from app.db import get_item
    from app.workers import classify

    item_id = _make_item_ready_for_classify(note="journal from March 2026")

    _stub_haiku(monkeypatch, {
        "path": "journal",
        "title": "March 15 entry",
        "one_line_summary": "Reflections on a hike.",
        "tags": ["journal", "hiking"],
        "date_of_content": "2026-03-15",
        "confidence": 0.9,
        "entities": {},
    })

    classify.process_one(item_id)

    row = get_item(item_id)
    assert row["path"] == "journal/2026/03"

    # File lives at journal/2026/03/<id>.jpg.
    assert (CONFIG.data_root / "journal" / "2026" / "03" / f"{item_id}.jpg").exists()


def test_classify_low_confidence_goes_to_inbox(tmp_data_root: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import CONFIG
    from app.db import get_item
    from app.workers import classify

    item_id = _make_item_ready_for_classify()

    # LLM is confident enough for records/financial but we're below floor.
    _stub_haiku(monkeypatch, {
        "path": "records/financial",
        "title": "Maybe a bill?",
        "one_line_summary": "Unclear.",
        "tags": [],
        "date_of_content": None,
        "confidence": 0.3,
        "entities": {},
    })

    classify.process_one(item_id)

    row = get_item(item_id)
    assert row["path"] == "inbox"
    # Confidence is stored as-is (we don't rewrite what the LLM returned;
    # we only override the routing decision).
    assert row["confidence"] == pytest.approx(0.3)

    assert (CONFIG.data_root / "inbox" / f"{item_id}.jpg").exists()


def test_classify_invalid_json_marks_failed(tmp_data_root: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from app.db import get_item
    from app.workers import classify

    item_id = _make_item_ready_for_classify()

    # Stub _call_haiku to raise directly, simulating a bad LLM response
    # that _parse_llm_json rejected.
    def broken(_transcript: str) -> dict:
        raise ValueError("LLM did not return valid JSON")

    monkeypatch.setattr(classify, "_call_haiku", broken)

    with pytest.raises(ValueError):
        classify.main()

    row = get_item(item_id)
    assert row["status"] == "failed"
    assert "LLM did not return valid JSON" in (row["error_message"] or "")


def test_classify_batch_pages_moved_alongside_raw(tmp_data_root: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """If the item was a multi-page batch (inbox/image/<id>/ exists),
    the page images move to <dest>/<id>.pages/ so vision re-runs later
    can find them."""
    from app.config import CONFIG
    from app.db import init_db, insert_item, update_item
    from app.workers import classify
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    insert_item(
        item_id=item_id,
        source_kind="pdf",
        original_filename="batch.pdf",
        mime_type="application/pdf",
        size_bytes=1,
    )
    (CONFIG.data_root / "inbox" / "pdf" / f"{item_id}.pdf").write_bytes(b"%PDF")

    pages_dir = CONFIG.data_root / "inbox" / "image" / item_id
    pages_dir.mkdir(parents=True)
    (pages_dir / "page-01.jpg").write_bytes(b"a")
    (pages_dir / "page-02.jpg").write_bytes(b"b")

    processed = CONFIG.data_root / "processed" / "pdf"
    processed.mkdir(parents=True)
    (processed / f"{item_id}.txt").write_text("batch text")
    update_item(item_id, status="transcribed",
                transcript_path=f"processed/pdf/{item_id}.txt",
                transcript_source="claude-vision-batch")
    (CONFIG.data_root / "queue" / "classify" / item_id).touch()

    _stub_haiku(monkeypatch, {
        "path": "reference/legal",
        "title": "Multi-page brief",
        "one_line_summary": "A case brief across pages.",
        "tags": ["legal"],
        "date_of_content": None,
        "confidence": 0.85,
        "entities": {},
    })

    classify.process_one(item_id)

    # Raw PDF at its new home.
    assert (CONFIG.data_root / "reference" / "legal" / f"{item_id}.pdf").exists()

    # The per-page images moved alongside as <id>.pages/.
    moved_pages = CONFIG.data_root / "reference" / "legal" / f"{item_id}.pages"
    assert moved_pages.is_dir()
    assert (moved_pages / "page-01.jpg").exists()
    assert (moved_pages / "page-02.jpg").exists()

    # Inbox is empty of this item.
    assert not (CONFIG.data_root / "inbox" / "image" / item_id).exists()
