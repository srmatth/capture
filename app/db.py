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


# Fields the pipeline is allowed to update. Whitelist so a typo never
# writes to a bogus column.
_UPDATABLE = frozenset({
    "status", "error_message",
    "path", "final_filename", "title", "one_line_summary",
    "date_of_content", "confidence", "classifier_version",
    "transcript_path", "transcript_char_count", "transcript_source",
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


def record_move(item_id: str, from_path: str | None, to_path: str, reason: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO moves(item_id, moved_at, from_path, to_path, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (item_id, _now(), from_path, to_path, reason),
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
