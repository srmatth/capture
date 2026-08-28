"""Prompt-builder tests. Cheap and content-only — no LLM calls."""

from __future__ import annotations

import json


def test_prompt_contains_all_taxonomy_keys() -> None:
    from app.taxonomy import TAXONOMY, build_classify_prompt
    prompt = build_classify_prompt("some item text")
    for key in TAXONOMY:
        assert key in prompt


def test_prompt_bounds_item_text_length() -> None:
    from app.taxonomy import build_classify_prompt
    # Use a rare marker char that can't appear in the taxonomy text —
    # sentinel is a private-use unicode code point.
    marker = ""
    long_text = marker * 20000
    prompt = build_classify_prompt(long_text)
    # 8000 char cap in the template; anything more would be a bug.
    assert prompt.count(marker) <= 8000
    assert prompt.count(marker) > 0     # sanity — it wasn't stripped entirely


def test_prompt_contains_schema() -> None:
    from app.taxonomy import build_classify_prompt
    prompt = build_classify_prompt("hi")
    for field in ("path", "title", "confidence", "one_line_summary", "entities"):
        assert field in prompt
