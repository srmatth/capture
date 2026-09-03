"""Upload endpoints.

/upload — single file. Writes inbox/<kind>/<id>.<ext>, DB row, returns id.

/upload_batch — N image files representing one logical multi-page item.
Server stitches them into a single PDF via img2pdf and processes as a
single-item PDF thereafter. The transcribe worker checks for the
companion inbox/image/<id>/ directory and, if handwritten, sends all
pages to Claude vision as one message rather than one call per page.

/jobs/<id> — polling endpoint the phone hits until an item reaches
'embedded' or 'failed'.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import img2pdf
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from ulid import ULID

from ..config import CONFIG
from ..db import get_item, insert_item, reset_retry_state, save_book_link

router = APIRouter()

# Reject uploads whose mime type we don't know how to route. Better an
# explicit error at POST time than a mystery file rotting in inbox/.
_KIND_BY_MIME_PREFIX: tuple[tuple[str, str], ...] = (
    ("audio/", "audio"),
    ("image/", "image"),
    ("application/pdf", "pdf"),
    ("text/plain", "plain"),
)

_DEFAULT_EXT = {"audio": "m4a", "image": "jpg", "pdf": "pdf", "plain": "txt"}


def _classify_mime(mime: str) -> str | None:
    for prefix, kind in _KIND_BY_MIME_PREFIX:
        if mime == prefix or mime.startswith(prefix):
            return kind
    return None


def _ext_for(file: UploadFile, kind: str) -> str:
    name = file.filename or ""
    if "." in name:
        return name.rsplit(".", 1)[-1].lower()
    return _DEFAULT_EXT[kind]


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    note: Annotated[str | None, Form()] = None,
    book_id: Annotated[int | None, Query()] = None,
    reading_id: Annotated[int | None, Query()] = None,
    book_title: Annotated[str | None, Query()] = None,
) -> dict:
    kind = _classify_mime(file.content_type or "")
    if kind is None:
        raise HTTPException(415, f"unsupported mime type: {file.content_type!r}")

    item_id = str(ULID())
    ext = _ext_for(file, kind)
    inbox_dir = CONFIG.data_root / "inbox" / kind
    inbox_dir.mkdir(parents=True, exist_ok=True)
    target = inbox_dir / f"{item_id}.{ext}"

    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    insert_item(
        item_id=item_id,
        source_kind=kind,
        original_filename=file.filename,
        mime_type=file.content_type,
        size_bytes=target.stat().st_size,
        upload_note=note,
    )

    if book_id is not None and reading_id is not None and book_title:
        save_book_link(item_id, book_id, reading_id, book_title)

    return {"id": item_id, "status_url": f"/jobs/{item_id}"}


@router.post("/upload_text")
async def upload_text(
    body: Annotated[str, Form()],
    note: Annotated[str | None, Form()] = None,
) -> dict:
    """Free-text capture. Text is both the raw content and the transcript.
    Skips the transcribe stage, goes straight to classify → embed."""
    if not body.strip():
        raise HTTPException(400, "text body is empty")

    item_id = str(ULID())
    inbox_dir = CONFIG.data_root / "inbox" / "plain"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    target = inbox_dir / f"{item_id}.txt"
    target.write_text(body)

    insert_item(
        item_id=item_id,
        source_kind="plain",
        original_filename="text-entry.txt",
        mime_type="text/plain",
        size_bytes=target.stat().st_size,
        upload_note=note,
    )
    return {"id": item_id, "status_url": f"/jobs/{item_id}"}


@router.post("/upload_url")
async def upload_url(
    url: Annotated[str, Form()],
    note: Annotated[str | None, Form()] = None,
) -> dict:
    """URL capture. Fetches the page, extracts readable text via
    readability-lxml, stores as a plain text item. The URL is saved
    in the upload_note for reference."""
    import ssl

    import httpx as _httpx
    from readability import Document

    ctx = ssl.create_default_context()
    ctx.load_default_certs()

    async with _httpx.AsyncClient(timeout=30, follow_redirects=True, verify=ctx) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    doc = Document(resp.text)
    title = doc.title() or ""
    content = doc.summary()

    # Strip HTML tags for a plain-text version
    from lxml import etree
    tree = etree.fromstring(content, parser=etree.HTMLParser())
    text = etree.tostring(tree, method="text", encoding="unicode").strip()

    if not text:
        raise HTTPException(400, "could not extract readable text from URL")

    item_id = str(ULID())
    inbox_dir = CONFIG.data_root / "inbox" / "plain"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    target = inbox_dir / f"{item_id}.txt"
    target.write_text(text)

    combined_note = f"url: {url}"
    if note:
        combined_note = f"{note}\nurl: {url}"

    insert_item(
        item_id=item_id,
        source_kind="plain",
        original_filename=title[:100] or "web-page.txt",
        mime_type="text/plain",
        size_bytes=target.stat().st_size,
        upload_note=combined_note,
    )
    return {"id": item_id, "status_url": f"/jobs/{item_id}"}


@router.post("/upload_batch")
async def upload_batch(
    files: list[UploadFile] = File(...),
    note: Annotated[str | None, Form()] = None,
) -> dict:
    """Multi-photo one-item upload.

    Every file must be an image. Server stitches them (in insertion order)
    into a single PDF via img2pdf, stores the originals in a per-item
    subdirectory so the transcribe worker can send them to Claude vision
    as a batch, and inserts one row with source_kind='pdf'.
    """
    if not files:
        raise HTTPException(400, "no files uploaded")
    for f in files:
        if not (f.content_type or "").startswith("image/"):
            raise HTTPException(
                415,
                f"batch upload accepts images only; got {f.content_type!r} "
                f"for {f.filename!r}",
            )

    item_id = str(ULID())

    # Preserve originals in inbox/image/<id>/ — the transcribe stage
    # detects this directory and treats the PDF as a stitched batch.
    pages_dir = CONFIG.data_root / "inbox" / "image" / item_id
    pages_dir.mkdir(parents=True, exist_ok=True)

    page_paths: list[Path] = []
    for i, f in enumerate(files, start=1):
        ext = _ext_for(f, "image")
        page_path = pages_dir / f"page-{i:02d}.{ext}"
        with page_path.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        page_paths.append(page_path)

    # Stitch into one PDF. img2pdf preserves image quality without
    # re-encoding (no JPEG-of-JPEG loss) and handles orientation.
    pdf_dir = CONFIG.data_root / "inbox" / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{item_id}.pdf"
    with pdf_path.open("wb") as out:
        out.write(img2pdf.convert([str(p) for p in page_paths]))

    insert_item(
        item_id=item_id,
        source_kind="pdf",
        original_filename=f"multi-page-{len(files)}pages.pdf",
        mime_type="application/pdf",
        size_bytes=pdf_path.stat().st_size,
        upload_note=note,
    )
    return {
        "id": item_id,
        "status_url": f"/jobs/{item_id}",
        "pages": len(files),
    }


@router.get("/jobs/{item_id}")
async def job_status(item_id: str) -> dict:
    row = get_item(item_id)
    if row is None:
        raise HTTPException(404)
    return {
        "id": row["id"],
        "status": row["status"],
        "error_message": row["error_message"],
        "title": row["title"],
        "path": row["path"],
        "retry_count": row["retry_count"],
        "last_error_at": row["last_error_at"],
    }


def _rewind_status(row: dict) -> str:
    """Figure out which pipeline stage a failed item belongs back at.

    Heuristic based on which output fields exist:
      - has path AND is 'dead_letter'/'failed' with path set → embed
        (classify already succeeded, only embed can be to blame)
      - has transcript_path but no path → classify
      - has no transcript_path → transcribe
    """
    if row.get("path"):
        return "classified"     # embed re-tries this
    if row.get("transcript_path"):
        return "transcribed"    # classify re-tries this
    return "queued"             # transcribe re-tries this


@router.post("/jobs/{item_id}/retry")
async def retry_job(item_id: str) -> dict:
    """Manual retry. Clears retry_count + last_error_at + error_message
    and rewinds status to the last-known-good stage so the appropriate
    worker picks it up on its next fire. Also touches the corresponding
    marker file so systemd path units notice immediately."""
    row = get_item(item_id)
    if row is None:
        raise HTTPException(404)
    if row["status"] not in ("failed", "dead_letter"):
        raise HTTPException(
            409,
            f"item is {row['status']!r}, only failed/dead_letter items can be retried",
        )

    new_status = _rewind_status(row)
    reset_retry_state(item_id, new_status)

    # Touch the correct marker so the path unit fires immediately.
    # Not strictly necessary since workers now consult the DB, but
    # keeps the systemd-fire loop tight.
    marker_dirs = {
        "queued": None,                                          # transcribe watches inbox/
        "transcribed": CONFIG.data_root / "queue" / "classify",
        "classified": CONFIG.data_root / "queue" / "embed",
    }
    marker_dir = marker_dirs[new_status]
    if marker_dir is not None:
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / item_id).touch()

    return {
        "id": item_id,
        "status": new_status,
        "retry_from_stage": new_status,
    }
