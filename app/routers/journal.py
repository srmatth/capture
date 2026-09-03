"""Journal compose endpoints.

Quick-compose a journal entry with text + optional photo attachments.
The entry flows through the normal capture pipeline (classify → embed)
with an upload_note hint so the classifier routes it to journal/.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from ulid import ULID

from ..config import CONFIG
from ..db import insert_attachment, insert_item, update_item

router = APIRouter()


def _templates():
    from ..main import TEMPLATES
    return TEMPLATES


@router.get("/journal/new", response_class=HTMLResponse)
async def journal_form(request: Request):
    return _templates().TemplateResponse(request, "journal.html")


@router.post("/journal/create")
async def journal_create(
    request: Request,
    body: str = Form(...),
    photos: list[UploadFile] = File(default=[]),
):
    item_id = str(ULID())
    now = datetime.now(timezone.utc).isoformat()

    inbox_dir = CONFIG.data_root / "inbox" / "plain"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    target = inbox_dir / f"{item_id}.txt"
    target.write_text(body)

    insert_item(
        item_id=item_id,
        source_kind="plain",
        original_filename="journal-entry.txt",
        mime_type="text/plain",
        size_bytes=len(body.encode("utf-8")),
        upload_note="journal",
    )

    # Attach any photos
    for photo in photos:
        if not photo.filename or not photo.size:
            continue
        att_id = str(ULID())
        ext = photo.filename.rsplit(".", 1)[-1].lower() if "." in photo.filename else "jpg"
        att_filename = f"{att_id}.{ext}"

        att_dir = CONFIG.data_root / "inbox" / f"{item_id}.attachments"
        att_dir.mkdir(parents=True, exist_ok=True)
        att_target = att_dir / att_filename

        with att_target.open("wb") as f:
            shutil.copyfileobj(photo.file, f)

        insert_attachment(
            attachment_id=att_id,
            item_id=item_id,
            filename=att_filename,
            mime_type=photo.content_type,
            size_bytes=att_target.stat().st_size,
        )

    return RedirectResponse(url=f"/jobs/{item_id}", status_code=303)
