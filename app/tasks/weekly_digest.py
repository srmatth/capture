"""Sunday-evening summary of the capture week.

Runs from the systemd unit capture-weekly-review.timer. Composes a
plain-text digest and pushes it to ntfy `me`. Includes:

- Filed cleanly per top-level category, with counts
- Items still in inbox/ (with count)
- Items filed with confidence 0.6-0.75 for spot-check
- Anything failed / dead-lettered this week
- Corpus size + weekly growth

Output goes to the ntfy `me` topic — this is a preview / briefing, not
an alerts-class notification. Uses the `NTFY_USER` / `NTFY_PASSWORD`
env from ntfy.env (same scripts credentials as the FreshRSS digest).

The task exits non-zero on push failure so systemd's OnFailure hook
fires. Empty-week runs still push (a quiet Sunday is still useful
signal — 'yes the pipeline ran, no you didn't upload anything').
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import httpx

from ..config import CONFIG

NTFY_URL = f"{os.environ.get('NTFY_BASE_URL', 'http://localhost:8080').rstrip('/')}/me"
NTFY_USER = os.environ.get("NTFY_USER", "scripts")
NTFY_PW = os.environ.get("NTFY_PASSWORD", "")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(CONFIG.library_db)
    conn.row_factory = sqlite3.Row
    return conn


def _top_level(path: str) -> str:
    """Group notes/personal, notes/professional, etc. under 'notes'."""
    return path.split("/", 1)[0] if path else "inbox"


def compose_digest(now: datetime | None = None) -> tuple[str, dict]:
    """Return (body_text, stats_dict). Stats returned separately so
    tests can assert on the numbers without regex-parsing text."""
    now = now or datetime.now(timezone.utc)
    week_start = now - timedelta(days=7)
    week_start_iso = week_start.isoformat()

    with _connect() as conn:
        # Items filed cleanly (or into inbox) this week. Uses
        # updated_at not uploaded_at so items processed asynchronously
        # after Sunday count in the week we actually classified them.
        filed_rows = conn.execute(
            "SELECT path FROM items "
            "WHERE deleted_at IS NULL AND path IS NOT NULL "
            "AND status IN ('classified', 'embedded') "
            "AND updated_at >= ?",
            (week_start_iso,),
        ).fetchall()

        counts: dict[str, int] = {}
        for r in filed_rows:
            top = _top_level(r["path"])
            counts[top] = counts.get(top, 0) + 1

        inbox_pending = conn.execute(
            "SELECT COUNT(*) FROM items "
            "WHERE deleted_at IS NULL AND path = 'inbox'"
        ).fetchone()[0]

        spot_check = conn.execute(
            "SELECT COUNT(*) FROM items "
            "WHERE deleted_at IS NULL AND path IS NOT NULL AND path <> 'inbox' "
            "AND confidence IS NOT NULL AND confidence < 0.75 "
            "AND updated_at >= ?",
            (week_start_iso,),
        ).fetchone()[0]

        failed = conn.execute(
            "SELECT status, COUNT(*) AS n FROM items "
            "WHERE deleted_at IS NULL "
            "AND status IN ('failed', 'dead_letter') "
            "AND updated_at >= ? "
            "GROUP BY status",
            (week_start_iso,),
        ).fetchall()
        failed_counts = {r["status"]: r["n"] for r in failed}

        # Corpus totals — all-time, not weekly.
        total = conn.execute(
            "SELECT COUNT(*) FROM items WHERE deleted_at IS NULL"
        ).fetchone()[0]

        week_uploaded = conn.execute(
            "SELECT COUNT(*) FROM items "
            "WHERE deleted_at IS NULL AND uploaded_at >= ?",
            (week_start_iso,),
        ).fetchone()[0]

    week_end = now
    header = (
        f"Week of {week_start.date().isoformat()} to {week_end.date().isoformat()}"
    )
    lines: list[str] = [header, "=" * len(header), ""]

    total_filed = sum(counts.values())
    if total_filed:
        lines.append(f"Filed: {total_filed} items")
        for top in sorted(counts):
            lines.append(f"  {top:<12} {counts[top]}")
        lines.append("")
    else:
        lines.append("Filed: 0 items this week")
        lines.append("")

    lines.append(f"Inbox needs review: {inbox_pending} items")
    lines.append(f"Low-confidence spot-checks: {spot_check} items")

    if failed_counts:
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(failed_counts.items()))
        lines.append(f"Failed / stuck: {parts}")
    else:
        lines.append("Failed / stuck: none")

    lines.append("")
    lines.append(f"Corpus: {total} items total, +{week_uploaded} uploaded this week")

    body = "\n".join(lines)
    stats = {
        "total_filed": total_filed,
        "per_category": counts,
        "inbox_pending": inbox_pending,
        "spot_check": spot_check,
        "failed": failed_counts,
        "corpus_total": total,
        "week_uploaded": week_uploaded,
    }
    return body, stats


def push_ntfy(body: str) -> None:
    resp = httpx.post(
        NTFY_URL,
        content=body.encode("utf-8"),
        headers={
            "Title": "Capture — weekly review",
            "Priority": "default",
            "Tags": "clipboard",
        },
        auth=(NTFY_USER, NTFY_PW) if NTFY_PW else None,
        timeout=10,
    )
    resp.raise_for_status()


def main() -> int:
    body, _stats = compose_digest()
    push_ntfy(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
