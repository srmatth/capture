"""Classify worker (Step 9).

Drains queue/classify/ markers left by the transcribe worker. For each
item: load the transcript, call Claude Haiku with the taxonomy prompt,
parse the response, decide the final path (with the confidence floor
kicking anything uncertain into inbox/), then move the raw file into
its destination and write a .meta.json alongside the transcript.

Handoff to the embed worker via a marker file in queue/embed/.

Every ledger-affecting decision writes to the `moves` audit table so
later prompt improvements can retrain against real human corrections.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from ..config import CONFIG
from ..db import (
    get_item,
    get_tags,  # noqa: F401 -- referenced elsewhere; import proves module wiring
    record_move,
    record_worker_failure,
    set_entities,
    set_tags,
    update_item,
)
from ..retry import is_ready_for_retry
from ..taxonomy import (
    CLASSIFIER_VERSION,
    CONFIDENCE_FLOOR,
    TAXONOMY,
    build_classify_prompt,
)

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Keys the LLM must return. Anything else in the response is ignored.
_REQUIRED_FIELDS = ("path", "title", "one_line_summary", "confidence")

# JSON fences the model sometimes wraps output in, despite the "no
# markdown fences" instruction.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _find_source_file(item_id: str, kind: str) -> Path:
    """Locate an item's raw file in inbox/. Same helper shape as the
    transcribe worker so a single file with any extension is found."""
    matches = sorted((CONFIG.data_root / "inbox" / kind).glob(f"{item_id}.*"))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise FileNotFoundError(f"no inbox file for {item_id}/{kind}")
    return matches[0]


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Strip any code fences and parse JSON. Raises ValueError on failure
    with a snippet of the offending text — the raw response is worth
    seeing when the model misbehaves."""
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}; got {cleaned[:300]!r}") from e
    missing = [k for k in _REQUIRED_FIELDS if k not in obj]
    if missing:
        raise ValueError(f"LLM response missing required fields: {missing}; got {obj!r}")
    return obj


def _call_haiku(transcript: str) -> dict[str, Any]:
    """Call Claude Haiku for classification. Returns the parsed JSON."""
    client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)
    prompt = build_classify_prompt(transcript)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        # System prompt is the taxonomy + rules text; user prompt is the
        # item text. Splitting them is what unlocks cache_control on the
        # system side so the taxonomy tokens get reused across items.
        system=[{
            "type": "text",
            "text": (
                "You are the librarian for a personal knowledge base. "
                "Follow the schema and rules in the user prompt exactly. "
                "Return only JSON, no markdown fences, no prose."
            ),
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content)
    return _parse_llm_json(text)


def _resolve_path(raw_path: str, confidence: float, item_id: str,
                   uploaded_at: str, date_of_content: str | None) -> str:
    """Turn the LLM's proposed path into the real on-disk destination.

    - Apply the confidence floor: anything below CONFIDENCE_FLOOR goes
      to inbox/ regardless of what the model said.
    - Journal path expands to journal/YYYY/MM/ using date_of_content
      when available, else the upload date. That's the one taxonomy
      key that gets temporal partitioning.
    - Unknown paths land in inbox/. The LLM sometimes hallucinates
      subfolders we didn't offer; safest to bail.
    """
    if confidence < CONFIDENCE_FLOOR:
        return "inbox"

    # Match against the taxonomy keys. Exact match wins; otherwise treat
    # as unknown and inbox it.
    if raw_path not in TAXONOMY:
        # Special case: the LLM might return 'notes/project/foo' for the
        # <name> placeholder. Accept the shape but only if the parent
        # notes/project/ is a real key (it is) — we adopt the leaf name
        # verbatim.
        if raw_path.startswith("notes/project/") and raw_path.count("/") == 2:
            leaf = raw_path.split("/", 2)[2]
            if leaf and leaf.replace("-", "").replace("_", "").isalnum():
                return raw_path
        return "inbox"

    # Temporal expansion for journal only.
    if raw_path == "journal":
        d = _parse_date(date_of_content) or _parse_date(uploaded_at)
        if d is not None:
            return f"journal/{d.year:04d}/{d.month:02d}"
        # Should never happen (uploaded_at is always set), but fall back
        # to journal/ root rather than crashing.
        return "journal"

    return raw_path


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    # Accept full ISO datetimes (uploaded_at) and bare dates
    # (date_of_content).
    try:
        return datetime.fromisoformat(s).date()
    except (TypeError, ValueError):
        try:
            return date.fromisoformat(s[:10])
        except (TypeError, ValueError):
            return None


def _extension_of(path: Path) -> str:
    return path.suffix.lstrip(".").lower() or "bin"


def process_one(item_id: str) -> None:
    row = get_item(item_id)
    if row is None:
        raise ValueError(f"item {item_id} not in DB")
    if not row["transcript_path"]:
        raise ValueError(f"item {item_id} has no transcript yet")

    transcript_full = (CONFIG.data_root / row["transcript_path"]).read_text()

    update_item(item_id, status="classifying")

    llm = _call_haiku(transcript_full)

    confidence = float(llm.get("confidence") or 0.0)
    final_path = _resolve_path(
        llm.get("path") or "inbox",
        confidence,
        row["id"],  # unused today but keeps a spot for future context
        row["uploaded_at"],
        llm.get("date_of_content"),
    )

    # Move the raw file to its destination folder.
    src = _find_source_file(item_id, row["source_kind"])
    dest_dir = CONFIG.data_root / final_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_filename = f"{item_id}.{_extension_of(src)}"
    dest_raw = dest_dir / final_filename
    shutil.move(str(src), str(dest_raw))

    # Multi-page batches have a companion inbox/image/<id>/ dir. Move
    # that alongside so we can retranscribe with Claude later without
    # asking the phone to re-upload.
    batch_pages = CONFIG.data_root / "inbox" / "image" / item_id
    if batch_pages.is_dir():
        shutil.move(str(batch_pages), str(dest_dir / f"{item_id}.pages"))

    # Write transcript + meta.json into parallel processed/ tree so the
    # destination folder stays browseable without transcript noise.
    processed_dir = CONFIG.data_root / "processed" / final_path
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_txt = processed_dir / f"{item_id}.txt"
    # Move rather than copy so we don't leave the old inbox-shaped
    # transcript in processed/<kind>/. The transcribe worker wrote it to
    # processed/<source_kind>/<id>.txt; relocate to processed/<path>/.
    old_txt = CONFIG.data_root / row["transcript_path"]
    if old_txt.exists() and old_txt.resolve() != processed_txt.resolve():
        shutil.move(str(old_txt), str(processed_txt))

    meta = {
        "id": item_id,
        "path": final_path,
        "title": (llm.get("title") or "").strip()[:120] or "(untitled)",
        "one_line_summary": (llm.get("one_line_summary") or "").strip()[:400],
        "date_of_content": llm.get("date_of_content"),
        "confidence": confidence,
        "tags": [str(t).strip().lower() for t in (llm.get("tags") or []) if str(t).strip()],
        "entities": llm.get("entities") or {},
        "classifier_version": CLASSIFIER_VERSION,
        "classified_at": datetime.now(timezone.utc).isoformat(),
    }
    (processed_dir / f"{item_id}.meta.json").write_text(json.dumps(meta, indent=2))

    # DB fields.
    update_item(
        item_id,
        status="classified",
        path=final_path,
        final_filename=final_filename,
        title=meta["title"],
        one_line_summary=meta["one_line_summary"],
        date_of_content=meta["date_of_content"],
        confidence=confidence,
        classifier_version=CLASSIFIER_VERSION,
        # transcript_path now points at the relocated file.
        transcript_path=str(processed_txt.relative_to(CONFIG.data_root)),
    )
    set_tags(item_id, meta["tags"])
    entities = meta["entities"]
    # Entities keys arrive as strings from the LLM; normalize the
    # container to str→list-of-str for set_entities.
    normalized_entities: dict[str, list[str]] = {}
    for k, v in entities.items():
        if isinstance(v, list):
            normalized_entities[str(k)] = [str(x) for x in v]
    set_entities(item_id, normalized_entities)

    # Record the placement in the moves audit so later prompt
    # improvements can measure "how often did the LLM's choice stick?"
    record_move(item_id, from_path=None, to_path=final_path, reason="classify")

    # Hand off to embed.
    embed_queue = CONFIG.data_root / "queue" / "embed"
    embed_queue.mkdir(parents=True, exist_ok=True)
    (embed_queue / item_id).touch()

    # Delete the classify marker last. If any of the above raised, the
    # marker stays and this item gets retried on the next fire.
    marker = CONFIG.data_root / "queue" / "classify" / item_id
    if marker.exists():
        marker.unlink()


def _pending_items() -> list[str]:
    """Return item IDs eligible for the classify stage. Includes:
    - 'transcribed' items with a fresh classify marker (normal path)
    - 'classifying' items (a previous run crashed after status update
      but before finishing) — these are retried
    - 'failed' items whose backoff window has elapsed and whose stage
      failure was during classify (identified by having transcript_path
      set but no path)

    Marker files are opportunistically recreated for retry-eligible
    items so path units downstream still see them, but we no longer
    require the marker's presence to run — the DB is authoritative.
    """
    from ..db import list_items_by_statuses

    queue = CONFIG.data_root / "queue" / "classify"
    queue.mkdir(parents=True, exist_ok=True)

    candidates = list_items_by_statuses(["transcribed", "classifying", "failed"])
    pending: list[str] = []
    for row in candidates:
        status = row["status"]
        if status in ("transcribed", "classifying"):
            pending.append(row["id"])
            continue
        # status == 'failed'
        # Only pick up failures that belong to THIS stage: transcript_path
        # set (transcribe succeeded) but path not yet set (classify never
        # completed).
        if not row["transcript_path"] or row["path"]:
            continue
        if is_ready_for_retry(row["retry_count"] or 0, row["last_error_at"]):
            pending.append(row["id"])

    # Sweep the queue dir for stale markers pointing at non-existent items.
    for marker in queue.iterdir():
        if get_item(marker.name) is None:
            marker.unlink()

    return pending


def main() -> int:
    pending = _pending_items()
    if not pending:
        return 0
    first_error: Exception | None = None
    for item_id in pending:
        try:
            process_one(item_id)
        except Exception as e:
            record_worker_failure(item_id, repr(e))
            if first_error is None:
                first_error = e
    if first_error is not None:
        raise first_error
    return 0


if __name__ == "__main__":
    sys.exit(main())
