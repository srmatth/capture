"""Canonical taxonomy + classification prompt."""

from __future__ import annotations

CLASSIFIER_VERSION = "haiku-4.5-taxonomy-v1"

TAXONOMY: dict[str, str] = {
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
    "inbox": (
        "Fallback when the item does not clearly fit any category. "
        "Return this with confidence <= 0.5 when uncertain."
    ),
}

CONFIDENCE_FLOOR = 0.6
"""Items classified below this confidence get force-routed to inbox/ regardless
of what the LLM said. Keeping the safety valve here rather than in the LLM
means we can adjust the threshold without a prompt change."""


def _taxonomy_lines() -> str:
    return "\n".join(f"- {path}: {desc}" for path, desc in TAXONOMY.items())


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
  "entities": {{
    "person": ["string", ...],
    "org": ["string", ...],
    "amount_usd": [number, ...]
  }}
}}

Item text (may be a transcript, OCR output, or plain text):
{item_text}
"""


def build_classify_prompt(item_text: str) -> str:
    return CLASSIFY_PROMPT_TEMPLATE.format(
        taxonomy_lines=_taxonomy_lines(),
        item_text=item_text[:8000],  # bound the prompt regardless of transcript length
    )
