"""Tasks Kanban endpoints.

A lightweight task board: quick-add, bucket into overdue / upcoming /
undated / snoozed, one-click done / snooze / delete.  The ntfy callback
endpoint lets push-notification action buttons mark tasks done or snooze
them without opening the browser.
"""

from __future__ import annotations

import os
import zoneinfo
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from ulid import ULID

from ..db import (
    delete_task,
    generate_reminder_token,
    get_task,
    get_task_by_reminder_token,
    get_tasks_for_project,
    insert_task,
    list_open_tasks,
    update_task,
)

router = APIRouter()

LOCAL_TZ = zoneinfo.ZoneInfo(os.environ.get("TZ", "America/Los_Angeles"))


def _templates():
    """Late-import to avoid a circular via main.py."""
    from ..main import TEMPLATES
    return TEMPLATES


def _now() -> str:
    return datetime.now(LOCAL_TZ).isoformat()


def _bucket_tasks(tasks: list[dict]) -> dict:
    """Split a flat task list into Kanban buckets."""
    now = datetime.now(LOCAL_TZ)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    overdue: list[dict] = []
    upcoming: list[dict] = []
    undated: list[dict] = []
    snoozed: list[dict] = []

    for t in tasks:
        if t["status"] == "snoozed":
            snoozed.append(t)
            continue
        if t["due_at"] is None:
            undated.append(t)
        else:
            try:
                due = datetime.fromisoformat(t["due_at"])
            except (ValueError, TypeError):
                undated.append(t)
                continue
            # Ensure timezone-aware comparison
            if due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            if due < now:
                overdue.append(t)
            else:
                upcoming.append(t)

    return {
        "overdue": overdue,
        "upcoming": upcoming,
        "undated": undated,
        "snoozed": snoozed,
    }


# ---------- views ----------


@router.get("/tasks", response_class=HTMLResponse)
async def tasks_kanban(request: Request) -> HTMLResponse:
    """Kanban view of all open + snoozed tasks."""
    tasks = list_open_tasks()
    buckets = _bucket_tasks(tasks)
    projects = sorted({t["project"] for t in tasks if t.get("project")})
    return _templates().TemplateResponse(
        request,
        "tasks.html",
        {
            **buckets,
            "project_filter": None,
            "projects": projects,
        },
    )


@router.get("/tasks/project/{slug}", response_class=HTMLResponse)
async def tasks_by_project(request: Request, slug: str) -> HTMLResponse:
    """Kanban view filtered to a single project."""
    tasks = get_tasks_for_project(slug)
    buckets = _bucket_tasks(tasks)
    projects = sorted({t["project"] for t in tasks if t.get("project")})
    return _templates().TemplateResponse(
        request,
        "tasks.html",
        {
            **buckets,
            "project_filter": slug,
            "projects": projects,
        },
    )


# ---------- mutations ----------


@router.post("/tasks/new")
async def task_create(
    title: Annotated[str, Form()],
    due_at: Annotated[str | None, Form()] = None,
    project: Annotated[str | None, Form()] = None,
    priority: Annotated[str, Form()] = "normal",
) -> RedirectResponse:
    """Quick-add a task from the Kanban form."""
    title = title.strip()
    if not title:
        raise HTTPException(400, "title is required")
    task_id = str(ULID())
    token = generate_reminder_token()
    insert_task(
        task_id=task_id,
        title=title,
        due_at=due_at if due_at else None,
        project=project.strip() if project else None,
        priority=priority if priority in ("high", "normal", "low") else "normal",
        source_item_id=None,
        reminder_token=token,
    )
    return RedirectResponse("/tasks", status_code=303)


@router.post("/tasks/{task_id}/done")
async def task_done(task_id: str) -> RedirectResponse:
    """Mark a task as done."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(404)
    update_task(task_id, status="done", completed_at=_now())
    return RedirectResponse("/tasks", status_code=303)


@router.post("/tasks/{task_id}/snooze")
async def task_snooze(
    task_id: str,
    hours: Annotated[int, Form()] = 24,
) -> RedirectResponse:
    """Snooze a task. Rotates the reminder token so any outstanding
    ntfy action buttons pointing at the old token become inert."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(404)
    until = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
    new_token = generate_reminder_token()
    update_task(
        task_id,
        status="snoozed",
        snoozed_until=until,
        reminder_token=new_token,
    )
    return RedirectResponse("/tasks", status_code=303)


@router.post("/tasks/{task_id}/edit")
async def task_edit(
    task_id: str,
    title: Annotated[str | None, Form()] = None,
    due_at: Annotated[str | None, Form()] = None,
    project: Annotated[str | None, Form()] = None,
    priority: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    """Edit task fields."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(404)
    fields: dict = {}
    if title is not None:
        title = title.strip()
        if title:
            fields["title"] = title
    if due_at is not None:
        fields["due_at"] = due_at if due_at else None
    if project is not None:
        fields["project"] = project.strip() if project.strip() else None
    if priority is not None and priority in ("high", "normal", "low"):
        fields["priority"] = priority
    if fields:
        update_task(task_id, **fields)
    return RedirectResponse("/tasks", status_code=303)


@router.post("/tasks/{task_id}/delete")
async def task_delete(task_id: str) -> RedirectResponse:
    """Delete a task permanently."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(404)
    delete_task(task_id)
    return RedirectResponse("/tasks", status_code=303)


# ---------- ntfy action callback ----------


@router.post("/tasks/action/{token}")
async def task_action(token: str, action: str = Query(...)) -> dict:
    """ntfy push-notification callback. The action buttons embed the
    reminder_token so the phone can fire done/snooze without auth."""
    task = get_task_by_reminder_token(token)
    if task is None:
        raise HTTPException(404, "token not found or already rotated")

    if action == "done":
        update_task(task["id"], status="done", completed_at=_now())
        return {"ok": True}
    elif action == "snooze":
        until = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        new_token = generate_reminder_token()
        update_task(
            task["id"],
            status="snoozed",
            snoozed_until=until,
            reminder_token=new_token,
        )
        return {"ok": True}
    else:
        raise HTTPException(400, f"unknown action {action!r}; expected 'done' or 'snooze'")
