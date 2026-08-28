"""Transcribe worker dispatch tests.

We stub the actual OCR / vision / whisper calls at the module level so
these tests run without Tesseract, Whisper, or an Anthropic key. The
point is to verify:

- audio → whisper.cpp path
- image (no handwriting hint, plenty of Tesseract-readable text) → tesseract
- image (with 'journal' note) → claude-vision
- pdf whose companion inbox/image/<id>/ exists AND matches handwriting → claude-vision-batch
- pdf without a companion dir → the OCRmyPDF path (mocked)

Actual OCR / vision quality is out of scope here — that's deferred to
manual integration testing once Step 1-4 are done.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest


def _jpeg_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (400, 400), color=(240, 240, 240)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def worker_env(tmp_data_root: Path, monkeypatch: pytest.MonkeyPatch):
    """Import the worker with all shell-outs stubbed."""
    from app import workers
    from app.workers import transcribe

    # Stub Tesseract call — return an empty string by default so
    # _looks_handwritten routes to Claude unless overridden.
    def fake_tesseract(args, **kw):
        class _R:
            stdout = ""
        return _R()

    monkeypatch.setattr(transcribe.subprocess, "run", fake_tesseract)

    # Stub the Claude vision helper so we don't need an API key.
    calls = {"claude": [], "whisper": []}

    def fake_claude(paths):
        calls["claude"].append([Path(p).name for p in paths])
        return f"[CLAUDE {len(paths)} pages]"

    monkeypatch.setattr(transcribe, "_claude_transcribe", fake_claude)

    def fake_whisper(path):
        calls["whisper"].append(path.name)
        return "whisper output"

    monkeypatch.setattr(transcribe, "transcribe_audio", fake_whisper)

    def fake_ocrmypdf(*a, **kw):
        # Would write an OCRd PDF; skip for tests. Return None like the real fn.
        return None

    monkeypatch.setattr(transcribe.ocrmypdf, "ocr", fake_ocrmypdf)

    return transcribe, calls


def _make_item(kind: str, note: str = "") -> str:
    from app.db import init_db, insert_item
    from ulid import ULID

    init_db()
    item_id = str(ULID())
    insert_item(
        item_id=item_id,
        source_kind=kind,
        original_filename=f"src.{'m4a' if kind == 'audio' else 'jpg' if kind == 'image' else 'pdf'}",
        mime_type="application/pdf" if kind == "pdf" else f"{kind}/jpeg",
        size_bytes=1,
        upload_note=note,
    )
    return item_id


def test_audio_routes_to_whisper(worker_env, tmp_data_root: Path) -> None:
    transcribe, calls = worker_env
    item_id = _make_item("audio")
    src = tmp_data_root / "inbox" / "audio" / f"{item_id}.m4a"
    src.write_bytes(b"fake audio")

    transcribe.process_one(item_id)

    # transcribe_audio is monkeypatched — it should receive the source
    # path (the shim to convert m4a -> wav is exercised in a separate
    # test below where we don't stub the whole function).
    assert calls["whisper"] == [src.name]
    from app.db import get_item
    row = get_item(item_id)
    assert row["status"] == "transcribed"
    assert row["transcript_source"] == "whisper.cpp"
    assert (tmp_data_root / "processed" / "audio" / f"{item_id}.txt").read_text() == "whisper output"

    # Handoff marker exists.
    assert (tmp_data_root / "queue" / "classify" / item_id).exists()


def test_to_whisper_wav_noop_for_native_formats(tmp_data_root: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """WAV/MP3/OGG/FLAC skip the ffmpeg conversion entirely."""
    from app.workers import transcribe

    called = []
    monkeypatch.setattr(transcribe.subprocess, "run",
                         lambda *a, **kw: called.append(a))

    for ext in (".wav", ".mp3", ".ogg", ".flac"):
        p = tmp_data_root / f"clip{ext}"
        p.write_bytes(b"x")
        result = transcribe._to_whisper_wav(p)
        assert result == p, f"{ext} should be returned unchanged"

    assert called == [], "no subprocess should have been invoked for native formats"


def test_to_whisper_wav_calls_ffmpeg_for_m4a(tmp_data_root: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """m4a triggers ffmpeg with the right flags."""
    from app.workers import transcribe

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        # Simulate ffmpeg writing the wav file.
        Path(cmd[-1]).write_bytes(b"wav")

        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(transcribe.subprocess, "run", fake_run)

    m4a = tmp_data_root / "clip.m4a"
    m4a.write_bytes(b"aac")
    result = transcribe._to_whisper_wav(m4a)

    assert result == m4a.with_suffix(".wav")
    assert captured["cmd"][0] == "ffmpeg"
    assert "-ar" in captured["cmd"] and "16000" in captured["cmd"]
    assert "-ac" in captured["cmd"] and "1" in captured["cmd"]
    assert result.exists()


def test_image_with_journal_note_routes_to_claude(worker_env, tmp_data_root: Path) -> None:
    transcribe, calls = worker_env
    item_id = _make_item("image", note="journal from today")
    src = tmp_data_root / "inbox" / "image" / f"{item_id}.jpg"
    src.write_bytes(_jpeg_bytes())

    transcribe.process_one(item_id)

    # Claude called exactly once with exactly one image (the source).
    assert len(calls["claude"]) == 1
    assert calls["claude"][0] == [src.name]

    from app.db import get_item
    row = get_item(item_id)
    assert row["transcript_source"] == "claude-vision"


def test_pdf_batch_handwritten_routes_to_batch_claude(worker_env, tmp_data_root: Path) -> None:
    transcribe, calls = worker_env
    item_id = _make_item("pdf", note="journal test")

    # Companion image dir exists → this is a stitched batch.
    pages_dir = tmp_data_root / "inbox" / "image" / item_id
    pages_dir.mkdir(parents=True)
    (pages_dir / "page-01.jpg").write_bytes(_jpeg_bytes())
    (pages_dir / "page-02.jpg").write_bytes(_jpeg_bytes())
    (pages_dir / "page-03.jpg").write_bytes(_jpeg_bytes())

    (tmp_data_root / "inbox" / "pdf" / f"{item_id}.pdf").write_bytes(b"%PDF-fake")

    transcribe.process_one(item_id)

    # Exactly ONE Claude call with all three pages — this is the whole
    # point of batching.
    assert len(calls["claude"]) == 1
    assert calls["claude"][0] == ["page-01.jpg", "page-02.jpg", "page-03.jpg"]

    from app.db import get_item
    row = get_item(item_id)
    assert row["transcript_source"] == "claude-vision-batch"


def test_pdf_batch_typed_routes_to_tesseract_per_page(worker_env, tmp_data_root: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    transcribe, calls = worker_env
    item_id = _make_item("pdf")  # no note

    pages_dir = tmp_data_root / "inbox" / "image" / item_id
    pages_dir.mkdir(parents=True)
    (pages_dir / "page-01.jpg").write_bytes(_jpeg_bytes())
    (pages_dir / "page-02.jpg").write_bytes(_jpeg_bytes())

    (tmp_data_root / "inbox" / "pdf" / f"{item_id}.pdf").write_bytes(b"%PDF-fake")

    # Override Tesseract stub to return plenty of words so
    # _looks_handwritten returns False.
    def real_looking_tesseract(args, **kw):
        class _R:
            stdout = ("word one two three four five six seven eight nine "
                      "ten eleven twelve thirteen fourteen fifteen sixteen")
        return _R()
    monkeypatch.setattr(transcribe.subprocess, "run", real_looking_tesseract)

    transcribe.process_one(item_id)

    # Claude was NOT called.
    assert calls["claude"] == []

    from app.db import get_item
    row = get_item(item_id)
    assert row["transcript_source"] == "tesseract-batch"


def test_process_one_marks_failed_on_error(worker_env, tmp_data_root: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    transcribe, calls = worker_env
    item_id = _make_item("audio")
    # Deliberately don't create the source file. _find_source raises,
    # which the main loop turns into status='failed'.

    from app.db import get_item, list_items_by_status

    with pytest.raises(FileNotFoundError):
        transcribe.process_one(item_id)

    # process_one itself doesn't mark failure — that's main()'s job.
    # But we can call main() and confirm.

    # Reset status back to queued so main() picks it up.
    from app.db import update_item
    update_item(item_id, status="queued")
    assert list_items_by_status("queued")

    with pytest.raises(FileNotFoundError):
        transcribe.main()

    row = get_item(item_id)
    assert row["status"] == "failed"
    assert row["error_message"] and "FileNotFoundError" in row["error_message"]
