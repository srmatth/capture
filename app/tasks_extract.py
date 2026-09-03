"""Task extraction from voice memo transcripts.

Called by the classify worker as a second-stage pass on items classified
under notes/*. Uses Claude Haiku to find action items in the transcript
and returns structured task dicts.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone

import anthropic

from .config import CONFIG

_log = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

TASK_EXTRACT_PROMPT = """You extract action items from a voice memo transcript.

Rules:
- Return valid JSON: {{"tasks": [ {{...}}, ... ]}}. Empty list is fine.
- title: <=80 chars, imperative mood ("Call Mom", not "Mom called").
- due_at: ISO 8601 date or datetime if explicitly stated, else null.
  Today is {today}. "tomorrow"/"thursday"/"next week" resolves relative.
- project: short lowercase slug if the note mentions a recurring
  project the user is working on (e.g. "schwab_barbell"), else null.
- priority: "high", "normal", or "low". Default "normal". Only mark
  "high" if the speaker uses urgency language ("urgent", "ASAP",
  "critical", "must", "immediately"). Only mark "low" if explicitly
  deprioritised ("when I get around to it", "no rush", "someday").
- Do NOT invent tasks. If the note is reflective/journaling, return {{"tasks": []}}.
- Do NOT split one action into multiple tasks.

Transcript:
{transcript}
"""


def extract_tasks(transcript: str, item_id: str) -> list[dict]:
    """Extract tasks from a transcript. Returns a list of task dicts
    with keys: title, due_at, project, priority."""
    if not transcript or len(transcript.strip()) < 20:
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = TASK_EXTRACT_PROMPT.format(
        today=today, transcript=transcript[:6000]
    )

    client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content)

    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        _log.warning("task extraction returned invalid JSON for item %s: %s",
                     item_id, cleaned[:200])
        return []

    tasks = obj.get("tasks", [])
    if not isinstance(tasks, list):
        return []

    if len(tasks) > 5:
        _log.warning(
            "task extraction returned %d tasks for item %s — likely mis-parse, "
            "dropping all", len(tasks), item_id
        )
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)
    validated: list[dict] = []
    for t in tasks:
        if not isinstance(t, dict) or not t.get("title"):
            continue
        title = str(t["title"])[:80]
        due_at = t.get("due_at")
        if due_at:
            try:
                dt = datetime.fromisoformat(str(due_at))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < cutoff:
                    due_at = None
            except (ValueError, TypeError):
                due_at = None

        priority = t.get("priority", "normal")
        if priority not in ("high", "normal", "low"):
            priority = "normal"

        validated.append({
            "title": title,
            "due_at": str(due_at) if due_at else None,
            "project": t.get("project") or None,
            "priority": priority,
        })

    return validated
