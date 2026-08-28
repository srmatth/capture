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
from ..db import get_item, list_items_by_status, update_item

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


def transcribe_audio(path: Path) -> str:
    """whisper.cpp on the audio file, English-only, medium model.

    Writes <input>.txt alongside the source via -otxt. -np suppresses the
    progress bar (systemd captures stdout, we don't want it filling the
    journal).
    """
    if not CONFIG.whisper_bin or not CONFIG.whisper_model:
        raise RuntimeError(
            "WHISPER_BIN / WHISPER_MODEL env vars not set — did Step 4 run?"
        )
    subprocess.run([
        str(CONFIG.whisper_bin), "-m", str(CONFIG.whisper_model),
        "-l", "en", "-otxt", "-np", "-f", str(path),
    ], check=True)
    return path.with_suffix(".txt").read_text()


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
    pending = list_items_by_status("queued")
    if not pending:
        return 0
    first_error: Exception | None = None
    for row in pending:
        try:
            process_one(row["id"])
        except Exception as e:
            update_item(row["id"], status="failed", error_message=repr(e))
            # Remember the first error but keep draining the rest — one
            # bad file shouldn't block a queue.
            if first_error is None:
                first_error = e
    if first_error is not None:
        # Non-zero exit so systemd sees the failure and OnFailure fires.
        raise first_error
    return 0


if __name__ == "__main__":
    sys.exit(main())
