"""Transcribe worker. Drains queued items → writes text to processed/,
updates DB, drops a marker in queue/classify/.

Batch-scan handling: for a PDF item whose sibling directory
inbox/image/<id>/ exists, the PDF was stitched from N phone photos
(see routers/upload.py::upload_batch). If handwriting is detected we
send ALL pages to Claude vision in one message rather than one call
per page — the model does better with the continuation context and it
halves the network chatter.

Invoked by systemd via `python -m app.workers.transcribe`. Every item
that fails to transcribe is marked status='failed' and the error is
re-raised so systemd's OnFailure=notify-fail@%n.service fires.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path

import anthropic
import ocrmypdf

from ..config import CONFIG
from ..db import (
    get_item,
    list_items_by_statuses,
    record_worker_failure,
    update_item,
)
from ..retry import is_ready_for_retry


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run with an up-front binary check.

    A bare subprocess.run(['pdftotext', ...]) fails with FileNotFoundError(2)
    both when the input file is missing AND when the executable itself isn't
    on PATH — same errno, different meaning, both look like 'the file is
    gone.' Distinguishing these two failure modes is worth two lines of
    defensive code. Called only for tools we shell out to (tesseract,
    pdftotext, ffmpeg, whisper.cpp)."""
    exe = cmd[0]
    if shutil.which(exe) is None and not Path(exe).is_file():
        raise RuntimeError(
            f"required executable {exe!r} not found on PATH — "
            f"is the package installed?"
        )
    return subprocess.run(cmd, **kwargs)

VISION_PROMPT = (
    "Transcribe the text in the image(s) verbatim. If there are multiple "
    "pages, transcribe in order, separated by '---page break---'. Preserve "
    "the author's line breaks and paragraph structure. If handwritten, "
    "transcribe as accurately as possible. Return only the transcription, "
    "no commentary."
)

# Keywords in the upload `note` field that force the Claude vision path.
# Users type these when they know Tesseract will do poorly.
#
# Two categories:
# - Handwriting/informal — always vision (Tesseract is hopeless).
# - Complex-layout hints — also vision, because Tesseract's psm=1
#   auto-segmentation fails on multi-column, magazine-style, and
#   scan-of-scan layouts. The vision model reads columns natively.
_HANDWRITING_HINT_KW = (
    "journal", "handwritten", "notebook", "diary", "letter", "note page",
)
_COMPLEX_LAYOUT_HINT_KW = (
    "column", "multi-column", "paper", "article",
    "textbook", "brief", "magazine", "newspaper",
)


# Tesseract page-segmentation mode 1 does its own layout analysis and
# outputs columns in reading order. Mode 6 assumed a single uniform block
# and read left-to-right across columns, producing salad on any
# multi-column document. psm=1 is the right default for real-world scans;
# psm=6 stays available as a fallback if psm=1 comes back suspiciously
# empty (rare, but has happened on very clean single-column single-page
# scans where auto-segmentation confidently decides there's nothing here).
_TESSERACT_PSM_PRIMARY = "1"
_TESSERACT_PSM_FALLBACK = "6"
_TESSERACT_MIN_WORDS = 20   # below this we retry with the fallback psm


def _tesseract_image(path: Path) -> str:
    """OCR one image. Tries psm=1 first, falls back to psm=6 if that
    comes back suspiciously short. Returns the better of the two."""
    primary = _run(
        ["tesseract", str(path), "-", "--psm", _TESSERACT_PSM_PRIMARY],
        capture_output=True, text=True, timeout=120,
    ).stdout or ""
    if len(primary.split()) >= _TESSERACT_MIN_WORDS:
        return primary
    fallback = _run(
        ["tesseract", str(path), "-", "--psm", _TESSERACT_PSM_FALLBACK],
        capture_output=True, text=True, timeout=120,
    ).stdout or ""
    # Return whichever produced more text.
    return fallback if len(fallback.split()) > len(primary.split()) else primary


# ---------- Audio ----------


# whisper.cpp's built-in miniaudio decoder only handles WAV, MP3, OGG,
# FLAC. iOS Voice Memos default to m4a (AAC-in-MP4), which fails silently
# at read_audio_data. We normalize everything to 16 kHz mono WAV first —
# also what Whisper expects internally, so this is free of quality cost.
_WHISPER_NATIVE_EXTS = {".wav", ".mp3", ".ogg", ".flac"}


def _to_whisper_wav(src: Path) -> Path:
    """Return a Path to a whisper-readable WAV. Cheap no-op if src is
    already one of Whisper's native formats."""
    if src.suffix.lower() in _WHISPER_NATIVE_EXTS:
        return src
    wav = src.with_suffix(".wav")
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-ar", "16000",     # Whisper's expected sample rate
        "-ac", "1",         # mono; Whisper collapses to mono internally anyway
        str(wav),
    ], check=True)
    return wav


def transcribe_audio(path: Path) -> str:
    """whisper.cpp on the audio file, English-only, medium model.

    Writes <input>.txt alongside the WAV via -otxt. -np suppresses the
    progress bar (systemd captures stdout, we don't want it filling
    the journal).
    """
    if not CONFIG.whisper_bin or not CONFIG.whisper_model:
        raise RuntimeError(
            "WHISPER_BIN / WHISPER_MODEL env vars not set — did Step 4 run?"
        )
    wav = _to_whisper_wav(path)
    _run([
        str(CONFIG.whisper_bin), "-m", str(CONFIG.whisper_model),
        "-l", "en", "-otxt", "-np", "-f", str(wav),
    ], check=True)
    # whisper.cpp -otxt writes <full-input-name>.txt (i.e. it appends
    # rather than replacing the extension). Path.with_suffix() replaces,
    # so we can't use it here.
    return Path(str(wav) + ".txt").read_text()


# ---------- Image / vision helpers ----------


def _looks_handwritten(img_path: Path) -> bool:
    """Cheap Tesseract sniff. If we find almost no real words, assume
    handwriting or a picture with no printed text and let the caller
    route to Claude. Uses psm=1 to match the main OCR path so a page
    that OCR won't handle at production time also fails here."""
    result = _run(
        ["tesseract", str(img_path), "-", "--psm", _TESSERACT_PSM_PRIMARY],
        capture_output=True, text=True, timeout=60,
    )
    words = [w for w in (result.stdout or "").split() if len(w) >= 3]
    return len(words) < 15


def _note_suggests_handwriting(note: str) -> bool:
    return any(k in (note or "").lower() for k in _HANDWRITING_HINT_KW)


def _note_suggests_complex_layout(note: str) -> bool:
    """The upload `note` may signal that the user knows this document
    has multi-column / magazine-style layout that trips OCR. When it
    does, prefer Claude vision over Tesseract regardless of the
    Tesseract confidence heuristic."""
    return any(k in (note or "").lower() for k in _COMPLEX_LAYOUT_HINT_KW)


def _should_use_vision(img_path: Path, note: str) -> bool:
    """Central routing decision — should this image go to Claude vision
    instead of Tesseract? Callers use this so the same logic applies
    across single-image, batch-image, and PDF-with-companion-images
    paths."""
    return (
        _note_suggests_handwriting(note)
        or _note_suggests_complex_layout(note)
        or _looks_handwritten(img_path)
    )


def _b64_image(path: Path) -> tuple[str, str]:
    ext = path.suffix.lstrip(".").lower()
    mime = f"image/{ext.replace('jpg', 'jpeg')}"
    return base64.b64encode(path.read_bytes()).decode(), mime


def _claude_transcribe(image_paths: list[Path]) -> str:
    """Send one or more images to Claude vision in a single message."""
    client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)
    content: list = []
    for p in image_paths:
        data, mime = _b64_image(p)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": data},
        })
    content.append({"type": "text", "text": VISION_PROMPT})
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": content}],
    )
    # Concatenate any text blocks — vision responses are usually a single
    # text block, but the API returns a list.
    return "".join(getattr(b, "text", "") for b in resp.content)


def transcribe_image(path: Path, note: str = "") -> tuple[str, str]:
    """Single-image transcribe. Returns (text, source_tag)."""
    if _should_use_vision(path, note):
        return _claude_transcribe([path]), "claude-vision"
    return _tesseract_image(path), "tesseract"


# Ratio below which a PDF's extracted text looks "too thin" for its
# file size, suggesting it's a hybrid (image body + thin text layer).
# Empirical: pure-text PDFs typically yield >100 chars/KB; scanned PDFs
# with a slim navigation-metadata layer come in well under 20.
_HYBRID_PDF_CHARS_PER_KB_THRESHOLD = 20


def _ocr_pdf_then_extract(pdf_path: Path, *, force_ocr: bool) -> str:
    """Run OCRmyPDF then pdftotext, return the extracted text.

    force_ocr=False is the fast path: OCRmyPDF skips pages that already
    have a text layer and only OCRs image-only pages. Right for the
    common case (scanned document with no existing text).

    force_ocr=True runs OCR on every page, discarding any existing
    text layer. This is the fix for 'hybrid' PDFs — printed-from-web
    pages that carry a slim text layer with nav/metadata boilerplate
    but embed the actual article body as an image.
    """
    suffix = ".force-ocr.pdf" if force_ocr else ".ocr.pdf"
    ocr_pdf = pdf_path.with_name(pdf_path.stem + suffix)
    ocrmypdf.ocr(
        pdf_path, ocr_pdf,
        force_ocr=force_ocr,
        skip_text=not force_ocr,
        language="eng",
        progress_bar=False,
    )
    return _run(
        ["pdftotext", "-layout", str(ocr_pdf), "-"],
        capture_output=True, text=True, timeout=180,
    ).stdout or ""


def _looks_like_hybrid_pdf(pdf_path: Path, extracted_text: str) -> bool:
    """Heuristic: was the initial extraction suspiciously thin for the
    file size? Hybrid PDFs (image body + thin text layer) have low
    chars-per-KB because pdftotext only saw the boilerplate."""
    try:
        size_kb = pdf_path.stat().st_size / 1024
    except OSError:
        return False
    if size_kb < 20:   # tiny PDFs are noise either way; skip the check
        return False
    return (len(extracted_text) / size_kb) < _HYBRID_PDF_CHARS_PER_KB_THRESHOLD


def transcribe_pdf(pdf_path: Path, item_id: str, note: str = "",
                    force_ocr: bool = False) -> tuple[str, str]:
    """PDF transcribe.

    If a companion inbox/image/<id>/ directory exists, this PDF is a
    stitched batch of phone photos. We use the individual images for
    the routing decision (Tesseract on an image is faster than running
    OCRmyPDF just to sniff), and if the routing says vision we send
    all pages to Claude in ONE call for better cross-page context.

    Otherwise (a genuine PDF upload), we run OCRmyPDF → pdftotext.
    pdftotext gets `-layout` so multi-column PDFs preserve their
    reading order — the equivalent of Tesseract's psm=1 for images.

    Two failure modes handled here:
    - Pure image PDFs (scans): OCRmyPDF with skip_text=True runs
      Tesseract on each page, adds a text layer, pdftotext reads it.
    - Hybrid PDFs (printed-from-website): the source has a thin text
      layer covering only nav/metadata boilerplate. skip_text sees the
      text and doesn't OCR, so we get "Home | About | Privacy" and no
      article body. After the first extraction we check the
      chars-per-KB ratio; below the threshold we re-run with
      force_ocr=True, which discards the existing text layer and
      re-OCRs everything.

    Caller can also pass force_ocr=True explicitly (used by the
    /item/<id>/retranscribe?with=force-ocr endpoint).
    """
    pages_dir = CONFIG.data_root / "inbox" / "image" / item_id
    if pages_dir.is_dir():
        pages = sorted(pages_dir.glob("page-*.*"))
        if pages:
            if _should_use_vision(pages[0], note):
                return _claude_transcribe(pages), "claude-vision-batch"
            parts = [_tesseract_image(p) for p in pages]
            return "\n---page break---\n".join(parts), "tesseract-batch"

    # Real PDF — OCRmyPDF path.
    if force_ocr:
        return _ocr_pdf_then_extract(pdf_path, force_ocr=True), "tesseract-pdf-forced"

    text = _ocr_pdf_then_extract(pdf_path, force_ocr=False)
    if _looks_like_hybrid_pdf(pdf_path, text):
        # Auto-fallback: text was suspiciously thin for the file size.
        # Re-run with force_ocr to bypass any existing text layer.
        # More expensive but correct.
        text = _ocr_pdf_then_extract(pdf_path, force_ocr=True)
        return text, "tesseract-pdf-forced"
    return text, "tesseract-pdf"


# ---------- Locating the raw file ----------


def _find_source(item_id: str, kind: str) -> Path:
    """Extension isn't always known (upload endpoint stores under the
    file's original ext). Glob for the id + any extension."""
    matches = sorted((CONFIG.data_root / "inbox" / kind).glob(f"{item_id}.*"))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise FileNotFoundError(f"no inbox file for {item_id}/{kind}")
    return matches[0]


# ---------- Main loop ----------


def _items_needing_retranscribe() -> list[dict]:
    """Fetch every item with retranscribe_hint set. Used by main() to
    pick up async retranscribe requests posted from the /retranscribe
    endpoint."""
    from ..db import _connect
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM items "
            "WHERE deleted_at IS NULL AND retranscribe_hint IS NOT NULL "
            "ORDER BY uploaded_at"
        ).fetchall()
        return [dict(r) for r in rows]


def _find_raw_for_reprocess(item_id: str, kind: str, row: dict) -> Path:
    """For an already-classified item, the raw file lives at
    <path>/<final_filename>, not in inbox/. Retranscribe needs to find it
    there. Falls through to the normal inbox lookup if path/filename
    aren't set (i.e., the item hasn't been classified yet)."""
    if row.get("path") and row.get("final_filename"):
        candidate = CONFIG.data_root / row["path"] / row["final_filename"]
        if candidate.exists():
            return candidate
    return _find_source(item_id, kind)


def process_one(item_id: str) -> None:
    row = get_item(item_id)
    if row is None:
        raise ValueError(f"item {item_id} not in DB")
    kind = row["source_kind"]
    note = row["upload_note"] or ""
    hint = row["retranscribe_hint"] or ""     # empty => normal routing

    # For a retranscribe request the raw file is under <path>/, not inbox/.
    if hint:
        src = _find_raw_for_reprocess(item_id, kind, row)
    else:
        src = _find_source(item_id, kind)

    update_item(item_id, status="transcribing")

    processed_dir = CONFIG.data_root / "processed" / kind
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_txt = processed_dir / f"{item_id}.txt"

    if hint:
        text, source_tag = _dispatch_hinted(hint, src, item_id, kind)
    elif kind == "audio":
        text, source_tag = transcribe_audio(src), "whisper.cpp"
    elif kind == "image":
        text, source_tag = transcribe_image(src, note)
    elif kind == "pdf":
        text, source_tag = transcribe_pdf(src, item_id, note)
    elif kind == "plain":
        text, source_tag = src.read_text(), "plaintext"
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    out_txt.write_text(text)
    update_item(
        item_id,
        status="transcribed",
        transcript_path=str(out_txt.relative_to(CONFIG.data_root)),
        transcript_char_count=len(text),
        transcript_source=source_tag,
        # Clear the hint so subsequent worker fires don't re-retranscribe.
        retranscribe_hint=None,
    )

    # Hand off to classify.
    queue = CONFIG.data_root / "queue" / "classify"
    queue.mkdir(parents=True, exist_ok=True)
    (queue / item_id).touch()


def _dispatch_hinted(hint: str, src: Path, item_id: str,
                      kind: str) -> tuple[str, str]:
    """Route a retranscribe request based on the explicit hint the API
    endpoint stashed in items.retranscribe_hint."""
    if hint == "vision":
        # For a batch upload, page images live under <path>/<id>.pages/.
        row = get_item(item_id) or {}
        path = row.get("path") or "inbox"
        batch_dir = CONFIG.data_root / path / f"{item_id}.pages"
        if batch_dir.is_dir():
            pages = sorted(batch_dir.glob("page-*.*"))
            if pages:
                return _claude_transcribe(pages), "claude-vision-batch"
        if kind == "image":
            return _claude_transcribe([src]), "claude-vision"
        raise RuntimeError(
            f"cannot vision-retranscribe kind={kind!r}: no source images available"
        )
    if hint == "tesseract":
        if kind == "image":
            return transcribe_image(src, note="")
        if kind == "pdf":
            return transcribe_pdf(src, item_id, note="")
        raise RuntimeError(f"cannot tesseract-retranscribe kind={kind!r}")
    if hint == "force-ocr":
        if kind != "pdf":
            raise RuntimeError(
                f"force-ocr is PDF-only, got kind={kind!r}"
            )
        return transcribe_pdf(src, item_id, note="", force_ocr=True)
    raise RuntimeError(f"unknown retranscribe hint {hint!r}")


def main() -> int:
    # Three sources of work:
    #   'queued'        fresh from upload
    #   'failed'        retry-eligible per app.retry backoff schedule
    #   retranscribe    items with retranscribe_hint set, regardless of status
    # Items in 'dead_letter' without a hint are terminal and never
    # picked up automatically.
    candidates = list_items_by_statuses(["queued", "failed"])
    pending = [
        row for row in candidates
        if row["status"] == "queued"
        or is_ready_for_retry(row["retry_count"] or 0, row["last_error_at"])
    ]
    # Retranscribe requests. A queued/failed item that ALSO has a hint
    # was already picked up above; skip dupes.
    pending_ids = {row["id"] for row in pending}
    for row in _items_needing_retranscribe():
        if row["id"] not in pending_ids:
            pending.append(row)
    if not pending:
        return 0
    first_error: Exception | None = None
    for row in pending:
        try:
            process_one(row["id"])
        except Exception as e:
            _, dead = record_worker_failure(row["id"], repr(e))
            # dead_letter items alert once (on transition). Retryable
            # failures alert every N tries by design — the OnFailure
            # notification firing on every retry is fine for a personal
            # setup and gives visibility while the item is stuck.
            if first_error is None:
                first_error = e
    if first_error is not None:
        raise first_error
    return 0


if __name__ == "__main__":
    sys.exit(main())
