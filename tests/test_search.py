"""Search / parsing / fusion tests. External services (Qdrant,
sentence-transformers) are stubbed at the function boundary via
monkeypatch — only the deterministic logic is exercised here."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------- parse_query ----------


def test_parse_query_extracts_known_operators(tmp_data_root) -> None:
    from app.search import parse_query
    got = parse_query("tag:antitrust after:2026-01 path:records case brief")
    assert got.filters == {
        "tag": ["antitrust"],
        "after": ["2026-01"],
        "path": ["records"],
    }
    assert got.text == "case brief"


def test_parse_query_supports_quoted_values(tmp_data_root) -> None:
    from app.search import parse_query
    got = parse_query('tag:"multi word" after:2026 anything')
    assert got.filters["tag"] == ["multi word"]
    assert got.filters["after"] == ["2026"]
    assert got.text == "anything"


def test_parse_query_unknown_operators_pass_through(tmp_data_root) -> None:
    """Typo like `tags:foo` (extra s) must not silently disappear."""
    from app.search import parse_query
    got = parse_query("tags:foo case brief")
    assert got.filters == {}
    assert "tags:foo" in got.text
    assert "case brief" in got.text


def test_parse_query_multi_valued_operator(tmp_data_root) -> None:
    """Two of the same operator should collect as a list."""
    from app.search import parse_query
    got = parse_query("tag:antitrust tag:legal something")
    assert set(got.filters["tag"]) == {"antitrust", "legal"}


def test_parse_query_empty(tmp_data_root) -> None:
    from app.search import parse_query
    got = parse_query("")
    assert got.filters == {}
    assert got.text == ""


# ---------- RRF ----------


def test_rrf_combines_two_lists_correctly(tmp_data_root) -> None:
    from app.search import reciprocal_rank_fusion, RRF_K
    a = [("x", 0.9), ("y", 0.8), ("z", 0.7)]  # ranks 1,2,3
    b = [("y", 100), ("w", 50)]                # ranks 1,2

    fused = reciprocal_rank_fusion([a, b])
    scores = dict(fused)

    # y appears in both at ranks 2 and 1.
    # x appears only in a at rank 1.
    # y gets 1/(K+2) + 1/(K+1); x gets 1/(K+1).
    # Since 1/(K+1) > 1/(K+2), the RRF math gives y > x.
    assert scores["y"] > scores["x"]
    assert scores["x"] > scores["z"]
    assert scores["z"] > 0
    assert scores["w"] > 0


def test_rrf_empty_lists_returns_empty(tmp_data_root) -> None:
    from app.search import reciprocal_rank_fusion
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# ---------- SQL filter compilation ----------


def test_sql_where_from_filters_tag_and_path(tmp_data_root) -> None:
    from app.search import _sql_where_from_filters
    where, params = _sql_where_from_filters(
        {"tag": ["antitrust"], "path": ["records"]}
    )
    assert "deleted_at IS NULL" in where
    assert "item_tags" in where
    assert "path = ?" in where or "path LIKE ?" in where
    assert "antitrust" in params
    assert "records" in params


def test_sql_where_from_filters_after_before(tmp_data_root) -> None:
    from app.search import _sql_where_from_filters
    where, params = _sql_where_from_filters(
        {"after": ["2026-01"], "before": ["2026-12-31"]}
    )
    assert "date_of_content >= ?" in where
    assert "date_of_content <= ?" in where
    assert "2026-01-01" in params
    assert "2026-12-31" in params


def test_sql_where_from_filters_ignores_bad_date(tmp_data_root) -> None:
    from app.search import _sql_where_from_filters
    where, params = _sql_where_from_filters({"after": ["not-a-date"]})
    # Unparseable date is silently dropped; no >= clause added.
    assert "date_of_content" not in where


# ---------- fts_search over real SQLite ----------


def _seed_item(item_id: str, title: str, summary: str,
                transcript_text: str, path: str = "notes/personal",
                tags: list[str] | None = None,
                date_of_content: str | None = None) -> None:
    """Insert an item plus a matching FTS row so keyword search finds it."""
    from app.db import (
        init_db, insert_item, set_tags, update_item, upsert_fts,
    )
    init_db()
    insert_item(item_id=item_id, source_kind="image",
                original_filename=f"{item_id}.jpg",
                mime_type="image/jpeg", size_bytes=1)
    update_item(item_id, status="embedded", path=path, title=title,
                one_line_summary=summary, date_of_content=date_of_content,
                confidence=0.9, classifier_version="v1",
                final_filename=f"{item_id}.jpg",
                transcript_path=f"processed/{path}/{item_id}.txt",
                transcript_char_count=len(transcript_text),
                transcript_source="tesseract")
    if tags:
        set_tags(item_id, tags)
    upsert_fts(item_id, title=title, summary=summary, transcript=transcript_text)


def test_fts_search_finds_keyword_matches(tmp_data_root: Path) -> None:
    from app.search import fts_search
    _seed_item("01FTS0000000000000000000A", title="Antitrust brief",
               summary="A summary about antitrust.",
               transcript_text="This document discusses antitrust litigation.")
    _seed_item("01FTS0000000000000000000B", title="Grocery list",
               summary="Just groceries.",
               transcript_text="Milk, eggs, bread.")

    hits = fts_search("antitrust")
    ids = [h[0] for h in hits]
    assert "01FTS0000000000000000000A" in ids
    assert "01FTS0000000000000000000B" not in ids


def test_fts_search_bad_query_returns_empty(tmp_data_root: Path) -> None:
    from app.search import fts_search
    _seed_item("01FTS0000000000000000000C", title="x", summary="x",
               transcript_text="x")
    # Unbalanced double-quote → OperationalError in SQLite FTS5.
    hits = fts_search('unmatched "quote')
    assert hits == []


# ---------- top-level search() with semantic stubbed ----------


def test_search_uses_fts_when_semantic_stubbed_empty(
    tmp_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With semantic search returning nothing, search() should still
    surface FTS matches."""
    from app import search as search_mod

    _seed_item("01SEA0000000000000000000A", title="Antitrust case brief",
               summary="Deep dive on antitrust.",
               transcript_text="antitrust antitrust antitrust")
    _seed_item("01SEA0000000000000000000B", title="Something unrelated",
               summary="Nope.",
               transcript_text="unrelated content here")

    monkeypatch.setattr(search_mod, "semantic_search", lambda t, limit=25: [])

    hits = search_mod.search("antitrust")
    ids = [h.item_id for h in hits]
    assert "01SEA0000000000000000000A" in ids
    assert "01SEA0000000000000000000B" not in ids


def test_search_filter_only_returns_recent_items(
    tmp_data_root: Path,
) -> None:
    """A query of just filters (no free text) should list matching items
    ordered by most-recent date_of_content."""
    from app import search as search_mod

    _seed_item("01FIL0000000000000000000A", title="Old",
               summary="", transcript_text="",
               tags=["antitrust"], date_of_content="2026-01-15")
    _seed_item("01FIL0000000000000000000B", title="New",
               summary="", transcript_text="",
               tags=["antitrust"], date_of_content="2026-08-15")
    _seed_item("01FIL0000000000000000000C", title="Not tagged",
               summary="", transcript_text="", tags=[])

    hits = search_mod.search("tag:antitrust")
    ids = [h.item_id for h in hits]
    assert ids[0] == "01FIL0000000000000000000B"
    assert ids[1] == "01FIL0000000000000000000A"
    assert "01FIL0000000000000000000C" not in ids


def test_soft_deleted_items_are_excluded(
    tmp_data_root: Path,
) -> None:
    """After a soft delete, search / browse / tag views must all hide
    the row."""
    from app.db import _connect
    from app.search import browse, list_by_tag, search

    _seed_item("01DEL0000000000000000000A", title="Antitrust brief",
               summary="", transcript_text="antitrust content",
               tags=["antitrust"], path="reference/legal")

    # Confirm it's initially findable.
    assert any(h.item_id == "01DEL0000000000000000000A"
                for h in list_by_tag("antitrust"))

    # Soft delete.
    with _connect() as conn:
        conn.execute("UPDATE items SET deleted_at = ? WHERE id = ?",
                     ("2026-08-28T12:00:00+00:00", "01DEL0000000000000000000A"))

    assert list_by_tag("antitrust") == []
    assert browse("reference/legal") == []
    assert search("tag:antitrust") == []
