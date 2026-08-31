"""HTTP surface for search, browse, and item detail.

Host-header aware: on capture.matthewshome the '/' route shows the
upload PWA (see main.py). On search.matthewshome the '/' route shows
the search box. The other routes here (browse, item detail, tags,
actions) work on both hosts since middleware doesn't need to gate them.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse


def _redirect_or_json(request: Request, url: str, payload: dict):
    """Browser form posts get a 303 redirect so the user lands on a
    usable page instead of a raw JSON blob. Programmatic callers that
    set Accept: application/json get the JSON body they expect."""
    accept = request.headers.get("accept", "").lower()
    if "application/json" in accept and "text/html" not in accept:
        return payload
    return RedirectResponse(url, status_code=303)

from ..config import CONFIG
from ..db import (
    _connect, get_item, get_tags, list_items_by_statuses, record_move,
    reset_retry_state, update_item,
)
from ..search import (
    browse as browse_items, list_by_tag, path_facets, related_items,
    search as run_search,
)
from ..taxonomy import get_taxonomy

router = APIRouter()


def _templates():
    """Late-import to avoid a circular via main.py."""
    from ..main import TEMPLATES
    return TEMPLATES


# ---------- landing ----------


@router.get("/search", response_class=HTMLResponse)
async def search_landing(request: Request, q: str = ""):
    """Combined search + recent-uploads view. Used both as the
    search.matthewshome home page (see main.py '/' route) and at
    /search on the capture host."""
    hits = run_search(q, limit=25) if q else []
    if not q:
        recent = _recent_items(limit=10)
    else:
        recent = []
    return _templates().TemplateResponse(
        request,
        "search.html",
        {
            "query": q,
            "hits": hits,
            "recent": recent,
            "facets": path_facets(),
        },
    )


def _recent_items(limit: int = 10):
    """Most recently classified items, for the landing page. Uses the
    hit shape so the template can render one row type."""
    from ..search import SearchHit
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM items WHERE deleted_at IS NULL "
            "AND status IN ('embedded', 'classified') "
            "ORDER BY uploaded_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        hits = []
        for r in rows:
            tags = [t["tag"] for t in conn.execute(
                "SELECT tag FROM item_tags WHERE item_id = ? ORDER BY tag",
                (r["id"],),
            )]
            hits.append(SearchHit(
                item_id=r["id"], title=r["title"] or "(untitled)",
                path=r["path"] or "", one_line_summary=r["one_line_summary"] or "",
                date_of_content=r["date_of_content"], tags=tags,
                source_kind=r["source_kind"], combined_score=0.0,
            ))
    return hits


# ---------- browse ----------


@router.get("/browse", response_class=HTMLResponse)
async def browse_view(request: Request, path: str = ""):
    """Folder-style browse. Root shows top-level facets; a specific
    path shows items under it plus the sub-taxonomy siblings."""
    if not path:
        return _templates().TemplateResponse(
            request,
            "browse_root.html",
            {"facets": path_facets(), "taxonomy": get_taxonomy()},
        )

    items = browse_items(path, limit=100)
    # Show child paths for navigation (one level down from `path`).
    prefix = f"{path}/"
    child_paths: set[str] = set()
    with _connect() as conn:
        for r in conn.execute(
            "SELECT DISTINCT path FROM items "
            "WHERE deleted_at IS NULL AND path LIKE ?",
            (f"{prefix}%",),
        ):
            rel = r["path"][len(prefix):]
            child_paths.add(prefix + rel.split("/", 1)[0])
    return _templates().TemplateResponse(
        request,
        "browse.html",
        {
            "path": path,
            "items": items,
            "child_paths": sorted(child_paths),
        },
    )


# ---------- tags ----------


@router.get("/tags/{tag}", response_class=HTMLResponse)
async def tag_view(request: Request, tag: str):
    items = list_by_tag(tag, limit=100)
    return _templates().TemplateResponse(
        request,
        "tag.html",
        {"tag": tag, "items": items},
    )


# ---------- item detail ----------


@router.get("/item/{item_id}", response_class=HTMLResponse)
async def item_detail(request: Request, item_id: str):
    row = get_item(item_id)
    if row is None:
        raise HTTPException(404)

    transcript = ""
    if row["transcript_path"]:
        t_path = CONFIG.data_root / row["transcript_path"]
        if t_path.exists():
            transcript = t_path.read_text()

    from ..db import get_comments
    return _templates().TemplateResponse(
        request,
        "item.html",
        {
            "row": dict(row),
            "tags": get_tags(item_id),
            "transcript": transcript,
            "related": related_items(item_id, limit=5),
            "taxonomy": sorted(get_taxonomy().keys()),
            "comments": get_comments(item_id),
        },
    )


# ---------- item actions ----------


@router.get("/item/{item_id}/raw")
async def item_raw(item_id: str):
    """Serve the item's raw file. Used by the item detail page for
    thumbnails / audio playback / PDF embeds AND as the 'download raw'
    action."""
    row = get_item(item_id)
    if row is None or row["deleted_at"]:
        raise HTTPException(404)
    if not row["path"] or not row["final_filename"]:
        raise HTTPException(409, "item not yet classified")
    raw = CONFIG.data_root / row["path"] / row["final_filename"]
    if not raw.exists():
        raise HTTPException(404, "raw file missing")
    return FileResponse(
        raw,
        media_type=row["mime_type"] or "application/octet-stream",
        filename=row["final_filename"],
    )


@router.post("/item/{item_id}/move")
async def item_move(request: Request, item_id: str, path: Annotated[str, Form()]):
    """Move an item's raw + transcript + meta.json to a new path.
    Updates DB and records the move in the audit table so retraining
    the classifier later can learn from real corrections."""
    row = get_item(item_id)
    if row is None or row["deleted_at"]:
        raise HTTPException(404)
    if not row["path"]:
        raise HTTPException(409, "item not yet classified — can't move an unfiled item")

    old_path = row["path"]
    if path == old_path:
        return _redirect_or_json(
            request, f"/item/{item_id}",
            {"moved": False, "reason": "already there"},
        )

    # Validate target: either a taxonomy key or a journal date subpath,
    # or a concrete notes/project/<leaf> — the template <name> only
    # exists in the taxonomy as a placeholder, but its concrete children
    # are legal move targets whether or not they're in the taxonomy yet.
    taxonomy = get_taxonomy()
    if path not in taxonomy and not path.startswith("journal/") \
            and not path.startswith("notes/project/"):
        raise HTTPException(400, f"unknown path {path!r}")

    src_raw = CONFIG.data_root / old_path / row["final_filename"]
    dest_dir = CONFIG.data_root / path
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_raw = dest_dir / row["final_filename"]

    if src_raw.exists():
        shutil.move(str(src_raw), str(dest_raw))

    # Batch pages dir travels with the raw file if it exists.
    src_pages = CONFIG.data_root / old_path / f"{item_id}.pages"
    if src_pages.is_dir():
        shutil.move(str(src_pages), str(dest_dir / f"{item_id}.pages"))

    # Transcript + meta.json live under processed/<path>/.
    new_transcript = None
    if row["transcript_path"]:
        old_t = CONFIG.data_root / row["transcript_path"]
        new_processed_dir = CONFIG.data_root / "processed" / path
        new_processed_dir.mkdir(parents=True, exist_ok=True)
        new_t = new_processed_dir / f"{item_id}.txt"
        if old_t.exists():
            shutil.move(str(old_t), str(new_t))
        new_transcript = str(new_t.relative_to(CONFIG.data_root))

        # Meta.json alongside the transcript.
        old_meta = old_t.with_suffix(".meta.json") if old_t.suffix == ".txt" \
                   else old_t.parent / f"{item_id}.meta.json"
        new_meta = new_processed_dir / f"{item_id}.meta.json"
        if old_meta.exists():
            meta = json.loads(old_meta.read_text())
            meta["path"] = path
            new_meta.write_text(json.dumps(meta, indent=2))
            old_meta.unlink()

    update_item(item_id, path=path,
                **({"transcript_path": new_transcript} if new_transcript else {}))
    record_move(item_id, from_path=old_path, to_path=path, reason="user")

    # Qdrant payload also carries path — update it so filtered search
    # on path stays accurate. Not fatal if Qdrant is down.
    try:
        from qdrant_client import QdrantClient
        from ..workers.embed import _ulid_to_uuid
        client = QdrantClient(url=CONFIG.qdrant_url)
        client.set_payload(
            collection_name="library",
            payload={"path": path},
            points=[_ulid_to_uuid(item_id)],
        )
    except Exception:
        pass

    return _redirect_or_json(
        request, f"/item/{item_id}",
        {"moved": True, "from": old_path, "to": path},
    )


_VALID_RETRANSCRIBE_HINTS = ("vision", "tesseract", "force-ocr")


@router.post("/item/{item_id}/retranscribe")
async def item_retranscribe(
    item_id: str,
    with_: str = Query("vision", alias="with"),
):
    """Queue a retranscribe. Returns immediately; the transcribe worker
    picks it up on the next fire (path units watch inbox/ so we also
    touch a marker there to trigger it right away). Downstream stages
    (classify, embed) chain automatically off the transcribe worker's
    handoff markers.

    Query string:
        ?with=vision      force Claude vision (default; the common case
                          when Tesseract clearly failed on a multi-column
                          or unusual-font document)
        ?with=tesseract   force local OCR (revert a vision transcript
                          that went wrong, or save API cost on a redo)
        ?with=force-ocr   PDF-only. Re-runs OCRmyPDF with force_ocr=True,
                          bypassing any existing text layer. The fix
                          for 'hybrid' PDFs printed from a website.

    Response is a job handle you can poll at /jobs/<id>.
    """
    row = get_item(item_id)
    if row is None or row["deleted_at"]:
        raise HTTPException(404)

    if with_ not in _VALID_RETRANSCRIBE_HINTS:
        raise HTTPException(400, f"with= must be one of {_VALID_RETRANSCRIBE_HINTS}")

    kind = row["source_kind"]
    if kind == "audio":
        raise HTTPException(409, "audio items are Whisper-only; no vision fallback")
    if kind == "plain":
        raise HTTPException(409, "plaintext items have nothing to re-transcribe")
    if with_ == "force-ocr" and kind != "pdf":
        raise HTTPException(
            409,
            "force-ocr is PDF-only; image items should use ?with=tesseract",
        )
    if with_ == "vision" and kind == "pdf":
        # Same guard as the sync version. Genuine (non-batch) PDFs don't
        # have companion page images to send to Claude.
        path_dir = CONFIG.data_root / (row["path"] or "inbox")
        batch_dir = path_dir / f"{item_id}.pages"
        if not batch_dir.is_dir():
            raise HTTPException(
                409,
                "vision retranscribe not supported for non-batch PDFs; "
                "the source page images aren't available.",
            )

    # Set the hint and rewind status so the worker picks the item up.
    # Do NOT clear existing path/final_filename — the raw file stays
    # where it is, and the retranscribe uses that location.
    update_item(item_id, retranscribe_hint=with_, status="queued")

    # Touch a marker in inbox/<kind>/ so the path unit fires immediately.
    # The transcribe worker itself checks the DB for retranscribe_hint,
    # not the marker location, so any inbox path unit fires the whole
    # scan — this is just to nudge systemd. If path units aren't running
    # (dev, manual test), the next `uv run python -m app.workers.transcribe`
    # will pick it up.
    ping = CONFIG.data_root / "inbox" / kind / f".retranscribe-{item_id}"
    ping.parent.mkdir(parents=True, exist_ok=True)
    ping.touch()

    return {
        "queued": True,
        "id": item_id,
        "with": with_,
        "status_url": f"/jobs/{item_id}",
    }


@router.post("/item/{item_id}/reclassify")
async def item_reclassify(request: Request, item_id: str):
    """Re-run the classify worker on this specific item. Uses the
    current prompt / model version — useful after taxonomy tweaks."""
    row = get_item(item_id)
    if row is None or row["deleted_at"]:
        raise HTTPException(404)
    if not row["transcript_path"]:
        raise HTTPException(409, "no transcript to reclassify")

    # The worker's process_one is idempotent — it moves the raw file
    # to the new destination, so calling it directly works.
    from ..workers import classify
    try:
        classify.process_one(item_id)
    except Exception as e:
        raise HTTPException(500, f"reclassify failed: {e!r}") from e

    updated = get_item(item_id)
    return _redirect_or_json(
        request, f"/item/{item_id}",
        {"reclassified": True, "path": updated["path"],
         "confidence": updated["confidence"]},
    )


@router.post("/item/{item_id}/delete")
async def item_delete(request: Request, item_id: str):
    """Soft delete. Sets deleted_at, hides from search/browse. Raw
    files stay on disk; a future Trash view can recover them.
    Browser form posts redirect back to the path the item was in
    (so the user sees the browse listing without the deleted row)
    or / if that's not known."""
    row = get_item(item_id)
    if row is None:
        raise HTTPException(404)
    if row["deleted_at"]:
        return _redirect_or_json(
            request, "/",
            {"deleted": False, "reason": "already deleted"},
        )
    from datetime import datetime, timezone
    with _connect() as conn:
        conn.execute(
            "UPDATE items SET deleted_at = ?, updated_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(),
             datetime.now(timezone.utc).isoformat(), item_id),
        )
    # Delete redirects to browse of the parent path — the item was
    # just removed from there, so it makes sense to see the updated
    # listing. Fall back to / for items that were never classified.
    redirect_to = f"/browse?path={row['path']}" if row["path"] else "/"
    return _redirect_or_json(
        request, redirect_to, {"deleted": True},
    )


@router.post("/item/{item_id}/undelete")
async def item_undelete(request: Request, item_id: str):
    row = get_item(item_id)
    if row is None:
        raise HTTPException(404)
    if not row["deleted_at"]:
        return _redirect_or_json(
            request, f"/item/{item_id}",
            {"undeleted": False, "reason": "not deleted"},
        )
    with _connect() as conn:
        conn.execute("UPDATE items SET deleted_at = NULL WHERE id = ?", (item_id,))
    return _redirect_or_json(
        request, f"/item/{item_id}", {"undeleted": True},
    )


# ---------- editable metadata ----------


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@router.post("/item/{item_id}/edit")
async def item_edit(
    request: Request,
    item_id: str,
    title: Annotated[str | None, Form()] = None,
    one_line_summary: Annotated[str | None, Form()] = None,
    date_of_content: Annotated[str | None, Form()] = None,
    tags: Annotated[str | None, Form()] = None,
):
    """Edit user-editable metadata. The LLM's original classification
    stays in the DB, but these fields can be overridden by the user —
    e.g., adding a publish date to a newspaper photo scan that doesn't
    carry one in the OCR text.

    All fields are optional; only submitted fields are updated. Passing
    an empty string clears the field (nullable columns become NULL).

    `tags` is comma-separated. Replaces the full tag set for the item.
    """
    row = get_item(item_id)
    if row is None or row["deleted_at"]:
        raise HTTPException(404)

    updates: dict[str, Any] = {}
    if title is not None:
        title = title.strip()
        if not title:
            raise HTTPException(400, "title must not be empty when provided")
        updates["title"] = title[:200]
    if one_line_summary is not None:
        updates["one_line_summary"] = one_line_summary.strip()[:400] or None
    if date_of_content is not None:
        v = date_of_content.strip()
        if v and not _DATE_RE.match(v):
            raise HTTPException(400, "date_of_content must be YYYY-MM-DD or empty")
        updates["date_of_content"] = v or None
    if updates:
        update_item(item_id, **updates)

    if tags is not None:
        from ..db import set_tags
        parts = [t.strip().lower() for t in tags.split(",")]
        set_tags(item_id, [t for t in parts if t])

    # Refresh FTS so title/summary/tag edits show up in keyword search
    # on the next hit. Uses the current transcript on disk.
    from ..db import upsert_fts
    updated = get_item(item_id)
    transcript_text = ""
    if updated["transcript_path"]:
        t_path = CONFIG.data_root / updated["transcript_path"]
        if t_path.exists():
            transcript_text = t_path.read_text()
    upsert_fts(
        item_id,
        title=updated["title"] or "",
        summary=updated["one_line_summary"] or "",
        transcript=transcript_text,
    )

    return _redirect_or_json(
        request, f"/item/{item_id}",
        {"edited": True, "fields": list(updates.keys()) + (["tags"] if tags is not None else [])},
    )


# ---------- comments ----------


@router.post("/item/{item_id}/comments")
async def item_add_comment(
    request: Request,
    item_id: str,
    body: Annotated[str, Form()],
):
    """Append a user comment to the item."""
    row = get_item(item_id)
    if row is None or row["deleted_at"]:
        raise HTTPException(404)
    body = body.strip()
    if not body:
        raise HTTPException(400, "comment must not be empty")

    from ..db import add_comment
    comment_id = add_comment(item_id, body)
    return _redirect_or_json(
        request, f"/item/{item_id}",
        {"added_comment": True, "id": comment_id},
    )


@router.post("/item/{item_id}/comments/{comment_id}/delete")
async def item_delete_comment(
    request: Request, item_id: str, comment_id: int,
):
    from ..db import delete_comment
    delete_comment(comment_id)
    return _redirect_or_json(
        request, f"/item/{item_id}",
        {"deleted_comment": True, "id": comment_id},
    )
