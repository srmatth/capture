"""Editable taxonomy admin page.

Manages the entries in the `taxonomy_entries` table. Built-in entries
(is_builtin=1) have their descriptions editable but can't be deleted;
user-added entries can be edited and deleted (as long as no items are
still filed under them).

Every change here immediately affects future classifications — the
classify worker re-reads the taxonomy on each item, and
`classifier_version()` hashes the sorted paths, so items get a
version string that reflects the taxonomy state at classify time.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..db import _connect
from ..taxonomy import (
    TaxonomyError,
    add_entry,
    delete_entry,
    edit_description,
    get_taxonomy_entries,
)

router = APIRouter()


def _templates():
    from ..main import TEMPLATES
    return TEMPLATES


def _item_counts() -> dict[str, int]:
    """Non-deleted item counts keyed by exact path. The admin page uses
    these both to inform edits and to enforce the delete guard visually
    (a user-added entry with items filed under it shows an inline hint
    that it can't be deleted yet)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT path, COUNT(*) AS n FROM items "
            "WHERE deleted_at IS NULL AND path IS NOT NULL "
            "GROUP BY path"
        ).fetchall()
    return {r["path"]: int(r["n"]) for r in rows}


@router.get("/taxonomy", response_class=HTMLResponse)
async def taxonomy_page(
    request: Request,
    flash: str | None = None,
    flash_kind: str = "info",
):
    """List every taxonomy entry with edit/delete controls. Includes an
    'add new' form at the top so a new category can be created without
    navigating away first."""
    entries = get_taxonomy_entries()
    counts = _item_counts()
    for e in entries:
        e["item_count"] = counts.get(e["path"], 0)
    flash_ctx = None
    if flash:
        flash_ctx = {
            "msg": flash,
            "kind": flash_kind if flash_kind in ("ok", "warn", "danger", "info") else "info",
        }
    return _templates().TemplateResponse(
        request,
        "taxonomy.html",
        {"entries": entries, "flash": flash_ctx},
    )


@router.post("/taxonomy/add")
async def taxonomy_add(
    path: Annotated[str, Form()],
    description: Annotated[str, Form()],
) -> RedirectResponse:
    try:
        add_entry(path.strip(), description)
    except TaxonomyError as e:
        return RedirectResponse(
            url=f"/taxonomy?flash={e}&flash_kind=danger",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/taxonomy?flash=added%20{path}&flash_kind=ok",
        status_code=303,
    )


@router.post("/taxonomy/{path:path}/edit")
async def taxonomy_edit(
    path: str,
    description: Annotated[str, Form()],
) -> RedirectResponse:
    try:
        edit_description(path, description)
    except TaxonomyError as e:
        return RedirectResponse(
            url=f"/taxonomy?flash={e}&flash_kind=danger",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/taxonomy?flash=updated%20{path}&flash_kind=ok",
        status_code=303,
    )


@router.post("/taxonomy/{path:path}/delete")
async def taxonomy_delete(path: str) -> RedirectResponse:
    try:
        delete_entry(path)
    except TaxonomyError as e:
        return RedirectResponse(
            url=f"/taxonomy?flash={e}&flash_kind=danger",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/taxonomy?flash=deleted%20{path}&flash_kind=ok",
        status_code=303,
    )
