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

VISION_PROMPT = (
    "Transcribe the text in the image(s) verbatim. If there are multiple "
    "pages, transcribe in order, separated by '---page break---'. Preserve "
    "the author's line breaks and paragraph structure. If handwritten, "
    "transcribe as accurately as possible. Return only the transcription, "
    "no commentary."
)

# Keywords in the upload `note` field that force the Claude vision path.
# Users type these when they know Tesseract will do poorly (handwriting,
# unusual fonts, scanned typewritten pages).
_HANDWRITING_HINT_KW = ("journal", "handwritten", "notebook", "diary", "letter", "note page")


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
    subprocess.run([
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
    subprocess.run([
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
    route to Claude."""
    result = subprocess.run(
        ["tesseract", str(img_path), "-", "--psm", "6"],
        capture_output=True, text=True, timeout=60,
    )
    words = [w for w in (result.stdout or "").split() if len(w) >= 3]
    return len(words) < 15


def _note_suggests_handwriting(note: str) -> bool:
    return any(k in (note or "").lower() for k in _HANDWRITING_HINT_KW)


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
    if _note_suggests_handwriting(note) or _looks_handwritten(path):
        return _claude_transcribe([path]), "claude-vision"
    text = subprocess.run(
        ["tesseract", str(path), "-", "--psm", "6"],
        capture_output=True, text=True, timeout=120,
    ).stdout
    return text, "tesseract"


def transcribe_pdf(pdf_path: Path, item_id: str, note: str = "") -> tuple[str, str]:
    """PDF transcribe.

    If a companion inbox/image/<id>/ directory exists, this PDF is a
    stitched batch of phone photos. We prefer the images for the
    handwriting decision (Tesseract on an image is faster and cheaper
    than running OCRmyPDF just to sniff), and if handwritten we send
    all pages to Claude vision as ONE call.

    Otherwise (a genuine PDF upload), we run OCRmyPDF → pdftotext.
    """
    pages_dir = CONFIG.data_root / "inbox" / "image" / item_id
    if pages_dir.is_dir():
        pages = sorted(pages_dir.glob("page-*.*"))
        if pages:
            handwritten = _note_suggests_handwriting(note) or _looks_handwritten(pages[0])
            if handwritten:
                return _claude_transcribe(pages), "claude-vision-batch"
            parts: list[str] = []
            for p in pages:
                text = subprocess.run(
                    ["tesseract", str(p), "-", "--psm", "6"],
                    capture_output=True, text=True, timeout=120,
                ).stdout
                parts.append(text)
            return "\n---page break---\n".join(parts), "tesseract-batch"

    # Real PDF — OCRmyPDF path.
    ocr_pdf = pdf_path.with_name(pdf_path.stem + ".ocr.pdf")
    ocrmypdf.ocr(pdf_path, ocr_pdf, force_ocr=False, skip_text=True,
                 language="eng", progress_bar=False)
    text = subprocess.run(
        ["pdftotext", str(ocr_pdf), "-"],
        capture_output=True, text=True, timeout=180,
    ).stdout
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


def process_one(item_id: str) -> None:
    row = get_item(item_id)
    if row is None:
        raise ValueError(f"item {item_id} not in DB")
    kind = row["source_kind"]
    note = row["upload_note"] or ""
    src = _find_source(item_id, kind)

    update_item(item_id, status="transcribing")

    processed_dir = CONFIG.data_root / "processed" / kind
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_txt = processed_dir / f"{item_id}.txt"

    if kind == "audio":
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
    )

    # Hand off to classify.
    queue = CONFIG.data_root / "queue" / "classify"
    queue.mkdir(parents=True, exist_ok=True)
    (queue / item_id).touch()


def main() -> int:
    # 'queued' = fresh from upload; 'failed' = retry-eligible per the
    # exponential-backoff schedule in app.retry. Items in 'dead_letter'
    # are terminal and never picked up automatically.
    candidates = list_items_by_statuses(["queued", "failed"])
    pending = [
        row for row in candidates
        if row["status"] == "queued"
        or is_ready_for_retry(row["retry_count"] or 0, row["last_error_at"])
    ]
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
