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
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from ulid import ULID

from ..config import CONFIG
from ..db import get_item, insert_item

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
    }
