"""Search / retrieval helpers.

Separate from routers/search.py because the search-fusion logic and
query parsing are worth unit-testing independent of HTTP wiring.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .config import CONFIG
from .db import _connect  # module-internal on purpose; tests can hit it too


# ---------- query parsing ----------


# Ordered list of (name, prefix, validator). Ordering matters only for
# consistent debugging output; the parser handles them in any order.
_OPERATORS = ("tag", "path", "before", "after", "type")

# Match operator:value at word boundaries. Value can be:
# - a quoted string: after:"2026-01-15"
# - or a bare word (no whitespace, no leading quote): tag:antitrust
_OP_RE = re.compile(
    r'(?<![A-Za-z0-9_])(?P<op>(?:' + "|".join(_OPERATORS) + r')):'
    r'(?:"(?P<qval>[^"]*)"|(?P<val>[^\s"]+))'
)


@dataclass
class ParsedQuery:
    """Structured form of a raw search input.

    filters carries operator=value pairs the caller applies as SQL /
    Qdrant filters. text is the residual free text — the part that
    becomes the semantic + FTS query."""
    text: str = ""
    filters: dict[str, list[str]] = field(default_factory=dict)

    def has_filters(self) -> bool:
        return any(self.filters.values())


def parse_query(raw: str) -> ParsedQuery:
    """Extract operator:value pairs; the rest is free text.

    Unknown operators pass through as literal query text — a typo like
    `tags:foo` (extra s) is safer to include in the semantic search
    than silently drop.
    """
    filters: dict[str, list[str]] = {}
    remaining = _OP_RE.sub(
        lambda m: _record_and_erase(m, filters), raw
    )
    return ParsedQuery(text=remaining.strip(), filters=filters)


def _record_and_erase(match: re.Match, into: dict[str, list[str]]) -> str:
    op = match.group("op").lower()
    val = match.group("qval") if match.group("qval") is not None else match.group("val")
    into.setdefault(op, []).append(val)
    return " "  # replace the whole `op:val` with a space so words on
               # either side don't get glued together


# ---------- filter -> SQL / Qdrant ----------


def _parse_date_filter(s: str) -> str | None:
    """Turn `2026-01` / `2026-01-15` / `2026` into an ISO date lower
    bound. Returns None if unparseable."""
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            d = datetime.strptime(s, fmt).date()
            return d.isoformat()
        except ValueError:
            continue
    return None


def _sql_where_from_filters(filters: dict[str, list[str]]) -> tuple[str, list[Any]]:
    """Build the WHERE snippet + params for the items table (no leading
    'WHERE'). Callers combine this with their own predicates."""
    clauses: list[str] = ["deleted_at IS NULL"]
    params: list[Any] = []

    for tag in filters.get("tag", []):
        clauses.append(
            "id IN (SELECT item_id FROM item_tags WHERE tag = ?)"
        )
        params.append(tag.lower())

    for path in filters.get("path", []):
        # Prefix match on path — path:records catches records/financial,
        # records/medical, etc.
        clauses.append("(path = ? OR path LIKE ?)")
        params.extend([path, f"{path}/%"])

    for after in filters.get("after", []):
        iso = _parse_date_filter(after)
        if iso:
            clauses.append("date_of_content >= ?")
            params.append(iso)

    for before in filters.get("before", []):
        iso = _parse_date_filter(before)
        if iso:
            clauses.append("date_of_content <= ?")
            params.append(iso)

    for kind in filters.get("type", []):
        clauses.append("source_kind = ?")
        params.append(kind.lower())

    return " AND ".join(clauses), params


# ---------- individual retrieval passes ----------


def semantic_search(text: str, limit: int = 25) -> list[tuple[str, float]]:
    """Return [(item_id, score), ...] via Qdrant. Empty list on any
    failure (missing model, missing collection) — search UI shouldn't
    500 because a lazy service isn't ready yet."""
    if not text:
        return []
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
    except ImportError:
        return []

    # Reuse module-level clients so successive requests share the
    # model. Lazy so import-time is fast for tests.
    global _MODEL, _QDRANT
    if _MODEL is None:
        _MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=str(CONFIG.hf_home) if CONFIG.hf_home else None,
        )
    if _QDRANT is None:
        _QDRANT = QdrantClient(url=CONFIG.qdrant_url)

    try:
        vector = _MODEL.encode(text).tolist()
        results = _QDRANT.search(
            collection_name="library",
            query_vector=vector,
            limit=limit,
        )
    except Exception:
        return []
    # Payload carries item_id (the ULID). The Qdrant point ID is a
    # ULID→UUID mapping, not the item_id itself.
    return [
        (r.payload.get("item_id"), float(r.score))
        for r in results
        if r.payload and r.payload.get("item_id")
    ]


_MODEL = None
_QDRANT = None


def fts_search(text: str, limit: int = 25) -> list[tuple[str, float]]:
    """Return [(item_id, bm25_score), ...] via SQLite FTS5. bm25 is
    negated (lower is a better match in FTS5), so we return -bm25 to
    make higher = better, matching semantic_search's convention."""
    if not text:
        return []
    with _connect() as conn:
        try:
            rows = conn.execute(
                "SELECT id, bm25(items_fts) AS score "
                "FROM items_fts WHERE items_fts MATCH ? "
                "ORDER BY score LIMIT ?",
                (text, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS query (unbalanced quotes, etc.). Rather
            # than 500, treat as no matches.
            return []
    return [(r["id"], -float(r["score"])) for r in rows]


# ---------- fusion ----------


RRF_K = 60  # standard constant, dampens the effect of high ranks


def reciprocal_rank_fusion(
    result_sets: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Combine ranked lists via RRF. Score for item i is
        sum_over_sets( weight[set] * 1 / (K + rank_in_set) )

    Items not present in a set contribute 0 from that set. Returns
    items sorted by combined score, descending. K=60 is the default
    from the original RRF paper — high enough to smooth over rank
    differences at the head, low enough to still favor top matches.
    """
    weights = weights or [1.0] * len(result_sets)
    combined: dict[str, float] = {}
    for i, results in enumerate(result_sets):
        w = weights[i]
        for rank, (item_id, _score) in enumerate(results):
            combined[item_id] = combined.get(item_id, 0.0) + w / (RRF_K + rank + 1)
    return sorted(combined.items(), key=lambda t: t[1], reverse=True)


# ---------- top-level search ----------


@dataclass
class SearchHit:
    item_id: str
    title: str
    path: str
    one_line_summary: str
    date_of_content: str | None
    tags: list[str]
    source_kind: str
    combined_score: float


def search(raw_query: str, limit: int = 25) -> list[SearchHit]:
    """One-shot search: parse the query, run semantic + FTS,
    fuse via RRF, apply filters, hydrate rows.

    Filters are applied at hydration time, not before fusion, so
    filter-only queries (all filters, no free text) still return
    results in a sensible order.
    """
    parsed = parse_query(raw_query)

    if parsed.text:
        semantic = semantic_search(parsed.text, limit=limit * 2)
        fts = fts_search(parsed.text, limit=limit * 2)
        fused = reciprocal_rank_fusion([semantic, fts])
    else:
        fused = []

    # Hydrate. If there's free text: fetch fused IDs and preserve fused
    # order. If it's a filter-only query: fetch by filter alone, ordered
    # by most recent.
    where_sql, params = _sql_where_from_filters(parsed.filters)

    with _connect() as conn:
        if fused:
            fused_ids = [item_id for item_id, _ in fused[:limit * 2]]
            placeholders = ",".join("?" * len(fused_ids))
            rows = conn.execute(
                f"SELECT * FROM items WHERE id IN ({placeholders}) AND {where_sql}",
                fused_ids + params,
            ).fetchall()
            by_id = {r["id"]: r for r in rows}
            hits = [
                _row_to_hit(by_id[item_id], score, conn)
                for item_id, score in fused
                if item_id in by_id
            ][:limit]
        else:
            rows = conn.execute(
                f"SELECT * FROM items WHERE {where_sql} "
                f"ORDER BY COALESCE(date_of_content, uploaded_at) DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            hits = [_row_to_hit(r, 0.0, conn) for r in rows]

    return hits


def _row_to_hit(row: sqlite3.Row, score: float, conn: sqlite3.Connection) -> SearchHit:
    tags = [t["tag"] for t in conn.execute(
        "SELECT tag FROM item_tags WHERE item_id = ? ORDER BY tag", (row["id"],),
    )]
    return SearchHit(
        item_id=row["id"],
        title=row["title"] or "(untitled)",
        path=row["path"] or "",
        one_line_summary=row["one_line_summary"] or "",
        date_of_content=row["date_of_content"],
        tags=tags,
        source_kind=row["source_kind"],
        combined_score=score,
    )


def browse(path_prefix: str, limit: int = 100) -> list[SearchHit]:
    """List items under a given path prefix, ordered by
    date_of_content desc. Used by the folder-style browse view."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM items WHERE deleted_at IS NULL "
            "AND (path = ? OR path LIKE ?) "
            "ORDER BY COALESCE(date_of_content, uploaded_at) DESC LIMIT ?",
            (path_prefix, f"{path_prefix}/%", limit),
        ).fetchall()
        return [_row_to_hit(r, 0.0, conn) for r in rows]


def list_by_tag(tag: str, limit: int = 100) -> list[SearchHit]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT i.* FROM items i "
            "JOIN item_tags t ON t.item_id = i.id "
            "WHERE t.tag = ? AND i.deleted_at IS NULL "
            "ORDER BY COALESCE(i.date_of_content, i.uploaded_at) DESC LIMIT ?",
            (tag.lower(), limit),
        ).fetchall()
        return [_row_to_hit(r, 0.0, conn) for r in rows]


def path_facets() -> list[tuple[str, int]]:
    """[(path, count)] for browse landing page. Groups by top-level
    directory so we see counts for journal/, notes/, etc."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT "
            "  CASE WHEN instr(path, '/') > 0 "
            "       THEN substr(path, 1, instr(path, '/') - 1) "
            "       ELSE path END AS top, "
            "  COUNT(*) AS n "
            "FROM items WHERE deleted_at IS NULL AND path IS NOT NULL "
            "GROUP BY top ORDER BY top"
        ).fetchall()
        return [(r["top"], r["n"]) for r in rows]


def related_items(item_id: str, limit: int = 5) -> list[SearchHit]:
    """Nearest-neighbor semantic matches for the given item, excluding
    the item itself. Empty list on any Qdrant failure."""
    from .workers.embed import _ulid_to_uuid

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return []

    global _QDRANT
    if _QDRANT is None:
        _QDRANT = QdrantClient(url=CONFIG.qdrant_url)

    try:
        point_uuid = _ulid_to_uuid(item_id)
        # Retrieve first to get the vector, then search on it.
        pts = _QDRANT.retrieve(
            collection_name="library",
            ids=[point_uuid],
            with_vectors=True,
        )
        if not pts or not pts[0].vector:
            return []
        results = _QDRANT.search(
            collection_name="library",
            query_vector=pts[0].vector,
            limit=limit + 1,  # +1 to compensate for self-match
        )
    except Exception:
        return []

    neighbor_ids = [
        r.payload.get("item_id") for r in results
        if r.payload and r.payload.get("item_id") and r.payload.get("item_id") != item_id
    ][:limit]
    if not neighbor_ids:
        return []
    placeholders = ",".join("?" * len(neighbor_ids))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM items WHERE id IN ({placeholders}) AND deleted_at IS NULL",
            neighbor_ids,
        ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [_row_to_hit(by_id[i], 0.0, conn) for i in neighbor_ids if i in by_id]
