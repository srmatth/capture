"""Editable taxonomy + classification prompt.

The taxonomy lives in the `taxonomy_entries` table (see migrations/006).
Built-in entries are seeded here on first startup — `seed_builtins()`
is idempotent via INSERT OR IGNORE, so adding a new built-in later just
requires updating _BUILTINS and it will land on next restart.

Every read goes through `get_taxonomy()`. There's no in-process cache:
prompts get rebuilt per classify call (cheap), and the DB is
authoritative across the worker + web processes.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .db import _connect, _now

# Built-in entries seeded on first startup. Once seeded these are
# protected from deletion (is_builtin=1) but descriptions remain
# editable — a description change re-tunes the classifier prompt.
_BUILTINS: dict[str, str] = {
    "journal": (
        "Personal reflection, thoughts, feelings. Handwritten pages from a "
        "notebook belong here. Journal entries are almost always filed under "
        "journal/YYYY/MM/ (the classify worker fills in the date subpath)."
    ),
    "notes/personal": (
        "Miscellaneous personal notes, ideas, sticky notes, quick jottings "
        "not part of a journal."
    ),
    "notes/professional": (
        "Work notes, meeting notes, project planning."
    ),
    "notes/project/<name>": (
        "Notes tied to a specific recurring project. Only use if the item "
        "clearly belongs to an established project the user has been "
        "working on. Do not invent project names."
    ),
    "reference/academic": (
        "Research papers, textbook excerpts, academic content."
    ),
    "reference/legal": (
        "Case briefs, statutes, contracts, legal analysis."
    ),
    "reference/technical": (
        "Technical documentation, tutorials, code references."
    ),
    "records/financial": (
        "Bills, bank/loan statements, tax documents, financial paperwork."
    ),
    "records/medical": (
        "Medical records, insurance documents, prescriptions."
    ),
    "records/property": (
        "Leases, deeds, warranties, home documents."
    ),
    "records/receipts": (
        "Transactional receipts."
    ),
    "media/articles": (
        "Newspaper photos, saved articles, news clippings."
    ),
    "media/podcasts": (
        "Podcast transcripts."
    ),
    "media/books": (
        "Book excerpts, passages, highlights, quotes from books the user "
        "is reading."
    ),
    "inbox": (
        "Fallback when the item does not clearly fit any category. "
        "Return this with confidence <= 0.5 when uncertain."
    ),
}

CONFIDENCE_FLOOR = 0.6
"""Items classified below this confidence get force-routed to inbox/ regardless
of what the LLM said. Keeping the safety valve here rather than in the LLM
means we can adjust the threshold without a prompt change."""


# Path validation for user-added entries. Lowercase alnum, hyphen,
# underscore; forward slashes for nesting; leaves must be non-empty.
# Explicitly *not* accepting the "<name>" template shape from users —
# that's a built-in-only pattern.
_PATH_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class TaxonomyError(ValueError):
    """Raised for invalid taxonomy operations (bad path, collision,
    deleting a built-in, deleting one with items filed under it)."""


def validate_path(path: str) -> None:
    """Raise TaxonomyError if `path` isn't a legal user-added taxonomy
    key. The <name> template + journal date-partitioning are handled
    by the classifier / router — user-added entries must be concrete."""
    if not path or path != path.strip():
        raise TaxonomyError("path must be non-empty and un-padded")
    segments = path.split("/")
    if not all(segments):
        raise TaxonomyError(f"path {path!r} has an empty segment")
    for seg in segments:
        if not _PATH_SEGMENT_RE.match(seg):
            raise TaxonomyError(
                f"segment {seg!r} in {path!r} must be lowercase alphanumeric "
                "with hyphens/underscores"
            )


def seed_builtins() -> None:
    """Idempotent seed of _BUILTINS into the taxonomy_entries table.
    Called from main.py at startup right after init_db(). Descriptions
    on rows that already exist are NOT overwritten — the user's edits
    win over the module-level defaults."""
    now = _now()
    with _connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO taxonomy_entries "
            "(path, description, is_builtin, created_at, updated_at) "
            "VALUES (?, ?, 1, ?, ?)",
            [(p, d, now, now) for p, d in _BUILTINS.items()],
        )


def get_taxonomy() -> dict[str, str]:
    """Return {path: description} for every taxonomy entry, ordered by path.
    This is what the classify prompt is built from and what the routers
    validate `path in taxonomy` against."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT path, description FROM taxonomy_entries ORDER BY path"
        ).fetchall()
    return {r["path"]: r["description"] for r in rows}


def get_taxonomy_entries() -> list[dict[str, Any]]:
    """Full rows (path, description, is_builtin, timestamps) for the
    admin page. Ordered by path so the UI groups top-level buckets."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT path, description, is_builtin, created_at, updated_at "
            "FROM taxonomy_entries ORDER BY path"
        ).fetchall()
    return [dict(r) for r in rows]


def _items_under_path(conn: sqlite3.Connection, path: str) -> int:
    """Count non-deleted items filed at exactly `path`. Used to guard
    delete. Does NOT descend children — the delete flow refuses only
    if the *specific* path is in use; renaming/moving items is a
    separate action the user can take first."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM items "
        "WHERE deleted_at IS NULL AND path = ?",
        (path,),
    ).fetchone()
    return int(row["n"] or 0)


def add_entry(path: str, description: str) -> None:
    """Create a new user-added taxonomy entry. Raises TaxonomyError on
    invalid path, collision, or empty description."""
    validate_path(path)
    description = description.strip()
    if not description:
        raise TaxonomyError("description must not be empty")
    now = _now()
    with _connect() as conn:
        try:
            conn.execute(
                "INSERT INTO taxonomy_entries "
                "(path, description, is_builtin, created_at, updated_at) "
                "VALUES (?, ?, 0, ?, ?)",
                (path, description, now, now),
            )
        except sqlite3.IntegrityError as e:
            raise TaxonomyError(f"path {path!r} already exists") from e


def edit_description(path: str, description: str) -> None:
    """Update the description on an existing entry. Both built-in and
    user-added entries are editable — descriptions are prompt-only, so
    changes never break existing filed items."""
    description = description.strip()
    if not description:
        raise TaxonomyError("description must not be empty")
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE taxonomy_entries SET description = ?, updated_at = ? "
            "WHERE path = ?",
            (description, _now(), path),
        )
        if cur.rowcount == 0:
            raise TaxonomyError(f"no taxonomy entry at {path!r}")


def delete_entry(path: str) -> None:
    """Delete a user-added entry. Refuses to delete built-ins, and
    refuses if any non-deleted items are filed at that path (user
    should move them first)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT is_builtin FROM taxonomy_entries WHERE path = ?",
            (path,),
        ).fetchone()
        if row is None:
            raise TaxonomyError(f"no taxonomy entry at {path!r}")
        if int(row["is_builtin"]) == 1:
            raise TaxonomyError(f"cannot delete built-in entry {path!r}")
        n_items = _items_under_path(conn, path)
        if n_items > 0:
            raise TaxonomyError(
                f"{n_items} items still filed under {path!r}; "
                "move them elsewhere before deleting"
            )
        conn.execute("DELETE FROM taxonomy_entries WHERE path = ?", (path,))


# --------------------------------------------------------------------------
# Prompt building. classifier_version now hashes the sorted paths so a
# taxonomy edit is reflected in the version string stored on each item —
# forensic queries and retrainings can honestly reproduce prompt state.
# --------------------------------------------------------------------------


def classifier_version() -> str:
    """Derive a stable version string from the current taxonomy state.
    Descriptions influence the prompt but the *set of paths* is what
    materially changes which categories exist, so we hash paths only.
    Changing wording doesn't invalidate prior classifications."""
    paths = sorted(get_taxonomy().keys())
    digest = hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()[:8]
    return f"haiku-4.5-taxonomy-{digest}"


def _taxonomy_lines(taxonomy: dict[str, str]) -> str:
    return "\n".join(f"- {path}: {desc}" for path, desc in taxonomy.items())


CLASSIFY_PROMPT_TEMPLATE = """You are the librarian for a personal knowledge base.
Classify the following item into exactly one path from the taxonomy below.

Taxonomy:
{taxonomy_lines}

Rules:
- Return valid JSON matching the schema. No prose, no markdown fences.
- If the item does not clearly belong to any specific category, return path="inbox" with confidence <= 0.5.
- confidence 0.0-1.0 reflects how sure you are of the path. If any doubt, err low.
- title: <= 80 chars, human-readable, useful in a search result. Never "Untitled".
- one_line_summary: <= 200 chars, plain English, no meta ("this document is about...").
- tags: 2-5 lowercase, hyphen-separated tags. Concrete concepts, not categories.
- date_of_content: ISO 8601 date if inferable from the text, else null.
- Do NOT invent facts. Only extract what is present.

Schema:
{{
  "path": "string, one of the taxonomy keys",
  "title": "string",
  "one_line_summary": "string",
  "tags": ["string", ...],
  "date_of_content": "YYYY-MM-DD or null",
  "confidence": 0.0-1.0,
  "book_id": "integer or null (only when path is media/books)",
  "entities": {{
    "person": ["string", ...],
    "org": ["string", ...],
    "amount_usd": [number, ...]
  }}
}}

Item text (may be a transcript, OCR output, or plain text):
{item_text}
"""


def build_classify_prompt(item_text: str, extra_context: str = "") -> str:
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(
        taxonomy_lines=_taxonomy_lines(get_taxonomy()),
        item_text=item_text[:8000],
    )
    if extra_context:
        prompt += f"\n\n{extra_context}\n"
    return prompt
