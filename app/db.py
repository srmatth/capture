"""SQLite access layer.

Every function opens a short-lived connection. Concurrency is minimal
(one FastAPI process, plus workers that each open their own connection
via the same module) and WAL mode handles the reader/writer overlap
cleanly. No SQLAlchemy — the schema is small and hand-written SQL is
easier to reason about here.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import CONFIG

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(CONFIG.library_db, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db() -> None:
    """Run any migration files not yet applied. Idempotent."""
    CONFIG.library_db.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS _migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            """
        )
        applied = {row["name"] for row in conn.execute("SELECT name FROM _migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            sql = path.read_text()
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO _migrations(name, applied_at) VALUES (?, ?)",
                (path.name, _now()),
            )


# ---------- items ----------


def insert_item(
    *,
    item_id: str,
    source_kind: str,
    original_filename: str | None,
    mime_type: str | None,
    size_bytes: int,
    upload_note: str | None = None,
) -> None:
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO items (
                id, uploaded_at, source_kind, original_filename, mime_type,
                size_bytes, status, updated_at, upload_note
            ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)
            """,
            (item_id, now, source_kind, original_filename, mime_type,
             size_bytes, now, upload_note),
        )


def get_item(item_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return dict(row) if row else None


def list_items_by_status(status: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM items WHERE status = ? ORDER BY uploaded_at",
            (status,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_items_by_statuses(statuses: list[str]) -> list[dict[str, Any]]:
    """Return items whose status is in the given list. Convenience for
    workers that pick up both 'ready-for-this-stage' rows AND retry-
    eligible 'failed' rows in one query."""
    if not statuses:
        return []
    placeholders = ",".join("?" * len(statuses))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM items WHERE status IN ({placeholders}) "
            "ORDER BY uploaded_at",
            statuses,
        ).fetchall()
        return [dict(r) for r in rows]


def record_worker_failure(item_id: str, error: str) -> tuple[int, bool]:
    """Called by a worker's exception handler. Bumps retry_count, sets
    last_error_at, decides between 'failed' (will retry) and
    'dead_letter' (terminal). Returns (new_retry_count, is_dead_letter)."""
    from .retry import should_dead_letter

    with _connect() as conn:
        row = conn.execute(
            "SELECT retry_count FROM items WHERE id = ?", (item_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"no item {item_id}")
        new_count = (row["retry_count"] or 0) + 1
        dead = should_dead_letter(new_count)
        conn.execute(
            "UPDATE items SET retry_count = ?, last_error_at = ?, "
            "status = ?, error_message = ?, updated_at = ? WHERE id = ?",
            (new_count, _now(), "dead_letter" if dead else "failed",
             error, _now(), item_id),
        )
    return new_count, dead


def reset_retry_state(item_id: str, new_status: str) -> None:
    """Manual retry: clear retry_count + last_error_at and rewind the
    row to `new_status`. The router chooses new_status based on which
    fields exist."""
    with _connect() as conn:
        conn.execute(
            "UPDATE items SET retry_count = 0, last_error_at = NULL, "
            "error_message = NULL, status = ?, updated_at = ? WHERE id = ?",
            (new_status, _now(), item_id),
        )


# Fields the pipeline is allowed to update. Whitelist so a typo never
# writes to a bogus column.
_UPDATABLE = frozenset({
    "status", "error_message",
    "path", "final_filename", "title", "one_line_summary",
    "date_of_content", "confidence", "classifier_version",
    "transcript_path", "transcript_char_count", "transcript_source",
    "retry_count", "last_error_at",
    "retranscribe_hint",
})


def update_item(item_id: str, **fields: Any) -> None:
    bad = set(fields) - _UPDATABLE
    if bad:
        raise ValueError(f"cannot update fields {bad}")
    if not fields:
        return
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [item_id]
    with _connect() as conn:
        conn.execute(f"UPDATE items SET {sets} WHERE id = ?", values)


# ---------- tags / entities ----------


def set_tags(item_id: str, tags: Iterable[str]) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM item_tags WHERE item_id = ?", (item_id,))
        conn.executemany(
            "INSERT INTO item_tags(item_id, tag) VALUES (?, ?)",
            [(item_id, t) for t in tags],
        )


def set_entities(item_id: str, entities: dict[str, list[str]]) -> None:
    """Replace all entities for an item. `entities` is like
    {'person': ['Alice'], 'org': ['Acme'], ...}."""
    with _connect() as conn:
        conn.execute("DELETE FROM item_entities WHERE item_id = ?", (item_id,))
        rows = [(item_id, kind, str(v)) for kind, vals in entities.items() for v in vals]
        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO item_entities(item_id, entity_type, entity_value) "
                "VALUES (?, ?, ?)",
                rows,
            )


def get_tags(item_id: str) -> list[str]:
    with _connect() as conn:
        return [r["tag"] for r in conn.execute(
            "SELECT tag FROM item_tags WHERE item_id = ? ORDER BY tag",
            (item_id,),
        )]


# ---------- moves audit ----------


# ---------- user-written comments ----------


def add_comment(item_id: str, body: str) -> int:
    """Append a comment. Returns the new comment's rowid."""
    body = body.strip()
    if not body:
        raise ValueError("comment body must not be empty")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO item_comments(item_id, body, created_at) "
            "VALUES (?, ?, ?)",
            (item_id, body, _now()),
        )
        return cur.lastrowid or 0


def get_comments(item_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, body, created_at FROM item_comments "
            "WHERE item_id = ? ORDER BY created_at",
            (item_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_comment(comment_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM item_comments WHERE id = ?", (comment_id,))


def record_move(item_id: str, from_path: str | None, to_path: str, reason: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO moves(item_id, moved_at, from_path, to_path, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (item_id, _now(), from_path, to_path, reason),
        )


# ---------- tasks ----------

import secrets


_TASK_UPDATABLE = frozenset({
    "title", "due_at", "project", "priority", "status",
    "completed_at", "snoozed_until", "reminder_token",
})

_PRIORITY_RANK = {"high": 0, "normal": 1, "low": 2}


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def generate_reminder_token() -> str:
    return secrets.token_urlsafe(16)


def insert_task(
    *,
    task_id: str,
    title: str,
    due_at: str | None = None,
    project: str | None = None,
    priority: str = "normal",
    source_item_id: str | None = None,
    reminder_token: str | None = None,
) -> None:
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO tasks "
            "(id, title, due_at, project, priority, status, source_item_id, "
            "reminder_token, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)",
            (task_id, title, due_at, project, priority,
             source_item_id, reminder_token, now, now),
        )


def get_task(task_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None


def list_open_tasks() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE status IN ('open', 'snoozed') "
            "ORDER BY "
            "CASE priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END, "
            "CASE WHEN due_at IS NULL THEN 1 ELSE 0 END, "
            "due_at ASC, created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_tasks_for_project(project: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE project = ? AND status IN ('open', 'snoozed') "
            "ORDER BY "
            "CASE priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END, "
            "CASE WHEN due_at IS NULL THEN 1 ELSE 0 END, "
            "due_at ASC, created_at ASC",
            (project,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_task(task_id: str, **fields: Any) -> None:
    bad = set(fields) - _TASK_UPDATABLE
    if bad:
        raise ValueError(f"cannot update task fields {bad}")
    if not fields:
        return
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [task_id]
    with _connect() as conn:
        conn.execute(f"UPDATE tasks SET {sets} WHERE id = ?", values)


def delete_task(task_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))


def is_duplicate_task(title: str, due_at: str | None) -> bool:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT title, due_at FROM tasks WHERE status = 'open'"
        ).fetchall()
    title_lower = title.lower().strip()
    for row in rows:
        existing_title = (row["title"] or "").lower().strip()
        dates_match = (due_at or "") == (row["due_at"] or "")
        if not dates_match:
            continue
        if title_lower == existing_title:
            return True
        if title_lower.startswith(existing_title) or existing_title.startswith(title_lower):
            return True
        if _levenshtein(title_lower, existing_title) <= 3:
            return True
    return False


def insert_task_alert(task_id: str, kind: str) -> bool:
    with _connect() as conn:
        try:
            conn.execute(
                "INSERT INTO task_alerts (task_id, kind, sent_at) VALUES (?, ?, ?)",
                (task_id, kind, _now()),
            )
            return True
        except Exception:
            return False


def get_task_alerts(task_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM task_alerts WHERE task_id = ?", (task_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_task_by_reminder_token(token: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE reminder_token = ?", (token,)
        ).fetchone()
        return dict(row) if row else None


# ---------- book links ----------


def save_book_link(
    item_id: str, book_id: int, reading_id: int, book_title: str
) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE items SET reading_book_id = ?, reading_reading_id = ?, "
            "reading_book_title = ?, updated_at = ? WHERE id = ?",
            (book_id, reading_id, book_title, _now(), item_id),
        )


# ---------- FTS ----------


# ---------- attachments ----------


def insert_attachment(
    *,
    attachment_id: str,
    item_id: str,
    filename: str,
    mime_type: str | None = None,
    size_bytes: int = 0,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO item_attachments "
            "(id, item_id, filename, mime_type, size_bytes, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (attachment_id, item_id, filename, mime_type, size_bytes, _now()),
        )


def get_attachments(item_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM item_attachments WHERE item_id = ? ORDER BY uploaded_at",
            (item_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_attachment(attachment_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "DELETE FROM item_attachments WHERE id = ?", (attachment_id,)
        )


# ---------- FTS ----------


def upsert_fts(item_id: str, *, title: str, summary: str, transcript: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM items_fts WHERE id = ?", (item_id,))
        conn.execute(
            "INSERT INTO items_fts(id, title, one_line_summary, transcript) "
            "VALUES (?, ?, ?, ?)",
            (item_id, title or "", summary or "", transcript or ""),
        )
