"""Inbox / review endpoints.

The inbox page shows everything that needs a human eye: items the LLM
routed to inbox/, low-confidence classifications, failed items, and
dead-letters. Batch actions (accept/move/delete) are HTMX-driven so a
single click on the phone doesn't reload the page.

The doc's Step 12 sketched this as its own page under /inbox.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from ..config import CONFIG
from ..db import _connect, get_item

router = APIRouter()

# Two thresholds. Below FLOOR the classify worker already forces
# path='inbox'. Below SPOT_CHECK it filed the item into a real path but
# the reviewer should still eyeball it.
SPOT_CHECK_THRESHOLD = 0.75


def _templates():
    from ..main import TEMPLATES
    return TEMPLATES


def _iso_seven_days_ago() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()


def _fetch_review_buckets() -> dict:
    """Bucket items needing review. Returns:
      needs_review: path='inbox' — the LLM couldn't place it
      spot_check:   filed elsewhere but confidence < 0.75
      failed:       status='failed' (will retry) or 'dead_letter' (won't)
    """
    with _connect() as conn:
        needs_review = [dict(r) for r in conn.execute(
            "SELECT * FROM items WHERE deleted_at IS NULL AND path = 'inbox' "
            "ORDER BY uploaded_at DESC"
        )]
        spot_check = [dict(r) for r in conn.execute(
            "SELECT * FROM items WHERE deleted_at IS NULL "
            "AND path IS NOT NULL AND path <> 'inbox' "
            "AND confidence IS NOT NULL AND confidence < ? "
            "AND status IN ('classified', 'embedded') "
            "ORDER BY uploaded_at DESC LIMIT 50",
            (SPOT_CHECK_THRESHOLD,),
        )]
        failed = [dict(r) for r in conn.execute(
            "SELECT * FROM items WHERE deleted_at IS NULL "
            "AND status IN ('failed', 'dead_letter') "
            "ORDER BY updated_at DESC LIMIT 50"
        )]
    return {
        "needs_review": needs_review,
        "spot_check": spot_check,
        "failed": failed,
    }


@router.get("/inbox", response_class=HTMLResponse)
async def inbox_page(request: Request):
    buckets = _fetch_review_buckets()
    from ..taxonomy import get_taxonomy
    return _templates().TemplateResponse(
        request,
        "inbox.html",
        {
            "buckets": buckets,
            "spot_check_threshold": SPOT_CHECK_THRESHOLD,
            "taxonomy": sorted(get_taxonomy().keys()),
        },
    )


# ---------- HTMX action fragments ----------
#
# Each fragment endpoint mutates the row and returns just the HTML for
# a "resolved" card. The client swaps the card in place with HTMX's
# hx-swap="outerHTML" so the page doesn't reload.


@router.post("/inbox/{item_id}/accept", response_class=HTMLResponse)
async def inbox_accept(request: Request, item_id: str):
    """Accept the current classification. For an inbox/ item this is
    a no-op filing (still inbox) — really only useful for spot-check
    items where the user is confirming the LLM got it right. We just
    mark the row 'reviewed' by writing to the moves audit as
    reason='user_accept', but don't move the file."""
    row = get_item(item_id)
    if row is None:
        raise HTTPException(404)
    from ..db import record_move
    record_move(item_id, from_path=row["path"], to_path=row["path"],
                reason="user_accept")
    return _card_html(request, item_id, resolved="accepted")


@router.post("/inbox/{item_id}/move", response_class=HTMLResponse)
async def inbox_move(
    request: Request,
    item_id: str,
    path: Annotated[str, Form()],
):
    """Delegate to the shared move endpoint's logic, then return the
    HTMX-shaped fragment for the card. Force the JSON-response path
    in item_move by asserting Accept: application/json on the way in."""
    from ..routers.search import item_move
    # A shallow scope clone with json-accepting headers so item_move
    # doesn't 303-redirect us back to the item detail page.
    fake_scope = dict(request.scope)
    headers = [(k, v) for k, v in request.headers.raw
                if k.lower() != b"accept"]
    headers.append((b"accept", b"application/json"))
    fake_scope["headers"] = headers
    fake_request = Request(fake_scope, request.receive)
    try:
        await item_move(fake_request, item_id, path)
    except HTTPException as e:
        raise e
    return _card_html(request, item_id, resolved=f"moved to {path}")


@router.post("/inbox/{item_id}/delete", response_class=HTMLResponse)
async def inbox_delete(request: Request, item_id: str):
    from ..routers.search import item_delete
    fake_scope = dict(request.scope)
    headers = [(k, v) for k, v in request.headers.raw
                if k.lower() != b"accept"]
    headers.append((b"accept", b"application/json"))
    fake_scope["headers"] = headers
    fake_request = Request(fake_scope, request.receive)
    await item_delete(fake_request, item_id)
    return _card_html(request, item_id, resolved="deleted")


@router.post("/inbox/{item_id}/retry", response_class=HTMLResponse)
async def inbox_retry(request: Request, item_id: str):
    """For failed/dead-letter items. Rewinds status, appropriate worker
    picks it up next fire."""
    from ..routers.upload import retry_job
    try:
        await retry_job(item_id)
    except HTTPException as e:
        raise e
    return _card_html(request, item_id, resolved="retry queued")


def _card_html(request: Request, item_id: str, resolved: str) -> HTMLResponse:
    """Minimal placeholder shown after an action succeeds. Keeping this
    as one small piece of Jinja rather than a whole template."""
    from markupsafe import escape
    html = (
        f'<article class="review-card resolved" '
        f'data-item-id="{escape(item_id)}">'
        f'<span class="badge">{escape(resolved)}</span>'
        f'</article>'
    )
    return HTMLResponse(html)
