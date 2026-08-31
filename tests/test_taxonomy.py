"""Taxonomy: DB helpers, prompt builder, HTTP admin routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------- prompt / builder ----------


def test_prompt_contains_all_seeded_taxonomy_keys(tmp_data_root: Path) -> None:
    """After startup, every built-in path from _BUILTINS should have
    been seeded into the DB and thus appear in the classifier prompt."""
    from app.db import init_db
    from app.taxonomy import _BUILTINS, build_classify_prompt, seed_builtins
    init_db()
    seed_builtins()
    prompt = build_classify_prompt("some item text")
    for key in _BUILTINS:
        assert key in prompt


def test_prompt_bounds_item_text_length(tmp_data_root: Path) -> None:
    from app.db import init_db
    from app.taxonomy import build_classify_prompt, seed_builtins
    init_db()
    seed_builtins()
    # Rare marker char that won't appear in the taxonomy text.
    marker = ""
    long_text = marker * 20000
    prompt = build_classify_prompt(long_text)
    assert prompt.count(marker) <= 8000
    assert prompt.count(marker) > 0


def test_prompt_contains_schema(tmp_data_root: Path) -> None:
    from app.db import init_db
    from app.taxonomy import build_classify_prompt, seed_builtins
    init_db()
    seed_builtins()
    prompt = build_classify_prompt("hi")
    for field in ("path", "title", "confidence", "one_line_summary", "entities"):
        assert field in prompt


def test_added_entry_appears_in_prompt(tmp_data_root: Path) -> None:
    """The prompt is DB-driven, so an added entry should appear in the
    next call to build_classify_prompt without a restart."""
    from app.db import init_db
    from app.taxonomy import add_entry, build_classify_prompt, seed_builtins
    init_db()
    seed_builtins()
    add_entry("reference/cooking", "Recipes and cookbook excerpts.")
    prompt = build_classify_prompt("hi")
    assert "reference/cooking" in prompt
    assert "Recipes and cookbook excerpts" in prompt


# ---------- classifier_version ----------


def test_classifier_version_stable_across_calls(tmp_data_root: Path) -> None:
    from app.db import init_db
    from app.taxonomy import classifier_version, seed_builtins
    init_db()
    seed_builtins()
    assert classifier_version() == classifier_version()


def test_classifier_version_changes_when_paths_change(tmp_data_root: Path) -> None:
    """Hash is over sorted paths — adding a new entry must change it."""
    from app.db import init_db
    from app.taxonomy import add_entry, classifier_version, seed_builtins
    init_db()
    seed_builtins()
    before = classifier_version()
    add_entry("reference/cooking", "Recipes.")
    after = classifier_version()
    assert before != after


def test_classifier_version_stable_across_description_edits(tmp_data_root: Path) -> None:
    """Editing a description is prompt-only — the hash keys off paths,
    so an edit should NOT invalidate the version. That means past items'
    classifier_version still matches the current path set."""
    from app.db import init_db
    from app.taxonomy import classifier_version, edit_description, seed_builtins
    init_db()
    seed_builtins()
    before = classifier_version()
    edit_description("journal", "Rewritten description for the journal category.")
    after = classifier_version()
    assert before == after


# ---------- validate_path ----------


@pytest.mark.parametrize("path", [
    "reference/cooking",
    "notes/professional/design-review",
    "media/podcasts",
    "records_extra",
])
def test_validate_path_accepts_normal_paths(tmp_data_root: Path, path: str) -> None:
    from app.taxonomy import validate_path
    validate_path(path)  # no raise


@pytest.mark.parametrize("path", [
    "",
    " ",
    "UPPER",
    "with spaces",
    "reference//double",
    "-leading-hyphen",
    "trailing/",
    "notes/project/<name>",   # template shape is built-in only
    "café",
])
def test_validate_path_rejects_bad_paths(tmp_data_root: Path, path: str) -> None:
    from app.taxonomy import TaxonomyError, validate_path
    with pytest.raises(TaxonomyError):
        validate_path(path)


# ---------- add / edit / delete ----------


def test_seed_builtins_is_idempotent(tmp_data_root: Path) -> None:
    from app.db import init_db
    from app.taxonomy import _BUILTINS, get_taxonomy_entries, seed_builtins
    init_db()
    seed_builtins()
    seed_builtins()  # second call: no error, no duplicates
    entries = get_taxonomy_entries()
    assert len(entries) == len(_BUILTINS)


def test_seed_preserves_user_edited_description(tmp_data_root: Path) -> None:
    """After a user edits a built-in description, restarting (which
    re-runs seed_builtins) must not clobber the edit."""
    from app.db import init_db
    from app.taxonomy import edit_description, get_taxonomy, seed_builtins
    init_db()
    seed_builtins()
    edit_description("journal", "MY CUSTOM WORDING")
    seed_builtins()
    assert get_taxonomy()["journal"] == "MY CUSTOM WORDING"


def test_add_entry_success(tmp_data_root: Path) -> None:
    from app.db import init_db
    from app.taxonomy import add_entry, get_taxonomy, seed_builtins
    init_db()
    seed_builtins()
    add_entry("reference/cooking", "Recipes and cookbook excerpts.")
    tx = get_taxonomy()
    assert tx["reference/cooking"] == "Recipes and cookbook excerpts."


def test_add_entry_collision_raises(tmp_data_root: Path) -> None:
    from app.db import init_db
    from app.taxonomy import TaxonomyError, add_entry, seed_builtins
    init_db()
    seed_builtins()
    with pytest.raises(TaxonomyError):
        add_entry("journal", "collision with built-in")


def test_add_entry_rejects_empty_description(tmp_data_root: Path) -> None:
    from app.db import init_db
    from app.taxonomy import TaxonomyError, add_entry, seed_builtins
    init_db()
    seed_builtins()
    with pytest.raises(TaxonomyError):
        add_entry("reference/cooking", "   ")


def test_edit_description_missing_path_raises(tmp_data_root: Path) -> None:
    from app.db import init_db
    from app.taxonomy import TaxonomyError, edit_description, seed_builtins
    init_db()
    seed_builtins()
    with pytest.raises(TaxonomyError):
        edit_description("does/not/exist", "new")


def test_delete_entry_user_added_success(tmp_data_root: Path) -> None:
    from app.db import init_db
    from app.taxonomy import add_entry, delete_entry, get_taxonomy, seed_builtins
    init_db()
    seed_builtins()
    add_entry("reference/cooking", "Recipes.")
    delete_entry("reference/cooking")
    assert "reference/cooking" not in get_taxonomy()


def test_delete_entry_refuses_builtin(tmp_data_root: Path) -> None:
    from app.db import init_db
    from app.taxonomy import TaxonomyError, delete_entry, seed_builtins
    init_db()
    seed_builtins()
    with pytest.raises(TaxonomyError):
        delete_entry("journal")


def test_delete_entry_refuses_when_items_filed_under_path(tmp_data_root: Path) -> None:
    """Even user-added paths can't be deleted if items are filed there."""
    from app.db import init_db, insert_item, update_item
    from app.taxonomy import (
        TaxonomyError, add_entry, delete_entry, seed_builtins,
    )
    init_db()
    seed_builtins()
    add_entry("reference/cooking", "Recipes.")
    insert_item(item_id="01T00000000000000000000001",
                source_kind="image", original_filename="x.jpg",
                mime_type="image/jpeg", size_bytes=1)
    update_item("01T00000000000000000000001",
                status="classified", path="reference/cooking")
    with pytest.raises(TaxonomyError):
        delete_entry("reference/cooking")


def test_delete_ignores_soft_deleted_items(tmp_data_root: Path) -> None:
    """A soft-deleted item at that path shouldn't block deletion —
    the guard only counts live items."""
    from app.db import _connect, init_db, insert_item, update_item
    from app.taxonomy import add_entry, delete_entry, get_taxonomy, seed_builtins
    init_db()
    seed_builtins()
    add_entry("reference/cooking", "Recipes.")
    insert_item(item_id="01T00000000000000000000002",
                source_kind="image", original_filename="x.jpg",
                mime_type="image/jpeg", size_bytes=1)
    update_item("01T00000000000000000000002",
                status="classified", path="reference/cooking")
    with _connect() as conn:
        conn.execute("UPDATE items SET deleted_at = '2026-08-31T00:00:00Z' "
                     "WHERE id = ?", ("01T00000000000000000000002",))
    delete_entry("reference/cooking")  # no raise
    assert "reference/cooking" not in get_taxonomy()


# ---------- HTTP: /taxonomy admin routes ----------


def _client():
    from app.main import app
    return TestClient(app)


def test_taxonomy_page_renders(tmp_data_root: Path) -> None:
    """GET /taxonomy lists all seeded entries."""
    r = _client().get("/taxonomy")
    assert r.status_code == 200
    # Built-ins visible.
    assert "journal" in r.text
    assert "notes/personal" in r.text
    # Add form present.
    assert '/taxonomy/add' in r.text


def test_taxonomy_add_via_post(tmp_data_root: Path) -> None:
    r = _client().post(
        "/taxonomy/add",
        data={"path": "reference/cooking",
              "description": "Recipes and cookbook excerpts."},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "flash=" in r.headers["location"]
    from app.taxonomy import get_taxonomy
    assert "reference/cooking" in get_taxonomy()


def test_taxonomy_add_bad_path_flashes_error(tmp_data_root: Path) -> None:
    r = _client().post(
        "/taxonomy/add",
        data={"path": "BAD PATH", "description": "irrelevant"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "flash_kind=danger" in r.headers["location"]


def test_taxonomy_edit_via_post(tmp_data_root: Path) -> None:
    r = _client().post(
        "/taxonomy/journal/edit",
        data={"description": "New wording"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    from app.taxonomy import get_taxonomy
    assert get_taxonomy()["journal"] == "New wording"


def test_taxonomy_delete_builtin_flashes_error(tmp_data_root: Path) -> None:
    r = _client().post("/taxonomy/journal/delete", follow_redirects=False)
    assert r.status_code == 303
    assert "flash_kind=danger" in r.headers["location"]
    from app.taxonomy import get_taxonomy
    assert "journal" in get_taxonomy()  # still there


def test_taxonomy_delete_user_added(tmp_data_root: Path) -> None:
    # Import order matters: get a client first so app.main runs init_db()
    # and seed_builtins() before we touch the taxonomy_entries table.
    client = _client()
    from app.taxonomy import add_entry, get_taxonomy
    add_entry("reference/cooking", "Recipes.")
    r = client.post(
        "/taxonomy/reference/cooking/delete", follow_redirects=False,
    )
    assert r.status_code == 303
    assert "reference/cooking" not in get_taxonomy()


def test_taxonomy_page_shows_item_count(tmp_data_root: Path) -> None:
    """The listing page annotates each entry with how many items are
    filed under it — used to show the delete-guard message."""
    client = _client()
    from app.db import insert_item, update_item
    from app.taxonomy import add_entry
    add_entry("reference/cooking", "Recipes.")
    insert_item(item_id="01T00000000000000000000010",
                source_kind="image", original_filename="x.jpg",
                mime_type="image/jpeg", size_bytes=1)
    update_item("01T00000000000000000000010",
                status="classified", path="reference/cooking")
    r = client.get("/taxonomy")
    assert r.status_code == 200
    # Count line for the cooking row — accept "1 item" or "1 items".
    assert "1 item" in r.text
