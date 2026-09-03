"""Check for due task reminders and push ntfy notifications.

Runs from the systemd unit capture-task-reminders.timer (every 15 min
or so). For each open task with a due_at, computes which reminder tiers
are now due, sends ntfy notifications, and records sent alerts so they
are not repeated.

Reminder tiers:
  7d      — 7 days before due_at
  1d      — 1 day before due_at
  morning — 08:00 America/Los_Angeles on the day of due_at
  due     — at due_at (or after, i.e. overdue)

Also un-snoozes tasks whose snoozed_until has passed.
"""

from __future__ import annotations

import logging
import os
import sys
import zoneinfo
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from ..db import (
    generate_reminder_token,
    get_task_alerts,
    insert_task_alert,
    list_open_tasks,
    update_task,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TIMEZONE = zoneinfo.ZoneInfo("America/Los_Angeles")

NTFY_BASE_URL = os.environ.get("NTFY_BASE_URL", "http://localhost:8080").rstrip("/")
NTFY_URL = f"{NTFY_BASE_URL}/me"
NTFY_USER = os.environ.get("NTFY_USER", "scripts")
NTFY_PW = os.environ.get("NTFY_PASSWORD", "")

CAPTURE_BASE_URL = os.environ.get(
    "CAPTURE_BASE_URL", "http://capture.matthewshome"
).rstrip("/")

# Tier offsets relative to due_at. "morning" is handled specially.
_TIER_OFFSETS: dict[str, timedelta] = {
    "7d": timedelta(days=7),
    "1d": timedelta(days=1),
    "due": timedelta(0),
}

_KIND_BODY: dict[str, str] = {
    "7d": "Due in 7 days",
    "1d": "Due tomorrow",
    "morning": "Due today",
    "due": "OVERDUE",
}


def _parse_due_at(raw: str) -> datetime:
    """Parse an ISO 8601 datetime, assuming UTC when no timezone is present."""
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _compute_due_alerts(task: dict, now: datetime) -> list[str]:
    """Return alert kinds that are currently due and not already sent."""
    raw = task.get("due_at")
    if not raw:
        return []

    due_at = _parse_due_at(raw)
    already_sent = {a["kind"] for a in get_task_alerts(task["id"])}

    due: list[str] = []

    # Fixed-offset tiers: 7d, 1d, due
    for kind, offset in _TIER_OFFSETS.items():
        if kind in already_sent:
            continue
        if now >= due_at - offset:
            due.append(kind)

    # Morning tier: 08:00 local on the day of due_at
    if "morning" not in already_sent:
        due_local = due_at.astimezone(TIMEZONE)
        morning_dt = due_local.replace(
            hour=8, minute=0, second=0, microsecond=0
        )
        if now >= morning_dt:
            due.append("morning")

    return due


def _send_reminder(task: dict, kind: str) -> bool:
    """Push a single reminder notification via ntfy. Returns True on success."""
    token = task.get("reminder_token", "")
    actions = "; ".join(
        [
            f"http, Done, {CAPTURE_BASE_URL}/tasks/action/{token}?action=done, method=POST, clear=true",
            f"http, Snooze 1d, {CAPTURE_BASE_URL}/tasks/action/{token}?action=snooze, method=POST, clear=true",
        ]
    )

    headers = {
        "Title": task["title"],
        "Priority": "5" if kind == "due" else "3",
        "Tags": "memo",
        "X-Actions": actions,
    }

    body = _KIND_BODY.get(kind, kind)

    try:
        resp = httpx.post(
            NTFY_URL,
            content=body.encode("utf-8"),
            headers=headers,
            auth=(NTFY_USER, NTFY_PW) if NTFY_PW else None,
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception:
        log.exception("Failed to send ntfy reminder for task %s kind=%s", task["id"], kind)
        return False


def _handle_snoozed(now: datetime) -> None:
    """Un-snooze tasks whose snoozed_until has passed."""
    for task in list_open_tasks():
        if task.get("status") != "snoozed":
            continue
        raw = task.get("snoozed_until")
        if not raw:
            continue
        snoozed_until = _parse_due_at(raw)
        if now >= snoozed_until:
            log.info("Un-snoozing task %s (%s)", task["id"], task["title"])
            update_task(
                task["id"],
                status="open",
                snoozed_until=None,
                reminder_token=generate_reminder_token(),
            )


def main() -> int:
    now = datetime.now(timezone.utc)

    _handle_snoozed(now)

    tasks = list_open_tasks()
    for task in tasks:
        if task.get("status") != "open":
            continue
        if not task.get("due_at"):
            continue

        kinds = _compute_due_alerts(task, now)
        for kind in kinds:
            if _send_reminder(task, kind):
                insert_task_alert(task["id"], kind)
                log.info("Sent %s reminder for task %s", kind, task["id"])

    # Optional Uptime Kuma heartbeat
    heartbeat_path = Path.home() / ".config/matthewshome/kuma-capture-task-reminders.url"
    if heartbeat_path.exists():
        url = heartbeat_path.read_text().strip()
        if url:
            try:
                httpx.get(url, timeout=10)
            except Exception:
                log.warning("Kuma heartbeat failed", exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
