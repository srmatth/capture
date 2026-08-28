"""Embed worker (Step 10).

Drains queue/embed/ markers. For each item: load transcript, embed with
sentence-transformers all-MiniLM-L6-v2 (384-dim, CPU-friendly), upsert
into Qdrant collection 'library' with a searchable payload, populate
items_fts for keyword search, mark status='embedded'.

The Qdrant collection is expected to already exist (created in Step 3
of PHASE_2_CAPTURE.md). We do NOT auto-create on first embed — that
would silently mask a misconfigured deployment.
"""

from __future__ import annotations

import sys
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from ..config import CONFIG
from ..db import (
    get_item,
    get_tags,
    list_items_by_statuses,
    record_worker_failure,
    update_item,
    upsert_fts,
)
from ..retry import is_ready_for_retry

# One-vector-per-item for personal-scale corpora. Chunking to sliding
# windows is on the deferred list — swap in when a specific search
# regression prompts it.
EMBED_MAX_CHARS = 2000

# Module-level singletons so successive process_one() calls in the same
# worker fire don't reload the model. First load is ~5s on this CPU;
# subsequent embeds are milliseconds.
_MODEL: SentenceTransformer | None = None
_QDRANT: QdrantClient | None = None


def _model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=str(CONFIG.hf_home) if CONFIG.hf_home else None,
        )
    return _MODEL


def _qdrant() -> QdrantClient:
    global _QDRANT
    if _QDRANT is None:
        _QDRANT = QdrantClient(url=CONFIG.qdrant_url)
    return _QDRANT


def _ulid_to_uuid(item_id: str) -> str:
    """Qdrant point IDs must be either an unsigned int or a UUID. ULIDs
    (Crockford base32) aren't valid UUIDs, so map them deterministically:
    ULID is 128 bits, UUID is 128 bits. Convert the ULID's base32
    encoding to bytes and reformat as a UUID. Deterministic + reversible
    if we ever need to."""
    # python-ulid ULIDs are 26-char Crockford base32. Import lazily so
    # tests can monkeypatch even if the lib isn't installed in the
    # subset environment.
    from ulid import ULID
    raw_bytes = ULID.from_str(item_id).bytes
    return str(uuid.UUID(bytes=raw_bytes))


def process_one(item_id: str) -> None:
    row = get_item(item_id)
    if row is None:
        raise ValueError(f"item {item_id} not in DB")
    if row["status"] not in ("classified", "embedding"):
        raise ValueError(
            f"item {item_id} status={row['status']!r}, expected classified"
        )
    if not row["transcript_path"]:
        raise ValueError(f"item {item_id} has no transcript")

    update_item(item_id, status="embedding")

    text = (CONFIG.data_root / row["transcript_path"]).read_text()
    truncated = text[:EMBED_MAX_CHARS]

    vector = _model().encode(truncated).tolist()

    tags = get_tags(item_id)
    _qdrant().upsert(
        collection_name="library",
        points=[PointStruct(
            id=_ulid_to_uuid(item_id),
            vector=vector,
            payload={
                "item_id": item_id,
                "title": row["title"] or "",
                "path": row["path"] or "",
                "tags": tags,
                "date_of_content": row["date_of_content"],
                "one_line_summary": row["one_line_summary"] or "",
                "uploaded_at": row["uploaded_at"],
            },
        )],
    )

    # Populate the FTS5 mirror for keyword search alongside vector search.
    upsert_fts(
        item_id,
        title=row["title"] or "",
        summary=row["one_line_summary"] or "",
        transcript=text,
    )

    update_item(item_id, status="embedded")

    # Delete our marker last so a mid-run crash re-runs cleanly.
    marker = CONFIG.data_root / "queue" / "embed" / item_id
    if marker.exists():
        marker.unlink()


def _pending_items() -> list[str]:
    """Return item IDs eligible for the embed stage. Same pattern as
    classify: includes normal 'classified' rows, in-flight 'embedding'
    rows, AND retry-eligible 'failed' rows whose stage-signature says
    they got stuck here (classify completed => path set, but status
    isn't 'embedded' yet)."""
    queue = CONFIG.data_root / "queue" / "embed"
    queue.mkdir(parents=True, exist_ok=True)

    candidates = list_items_by_statuses(["classified", "embedding", "failed"])
    pending: list[str] = []
    for row in candidates:
        status = row["status"]
        if status in ("classified", "embedding"):
            pending.append(row["id"])
            continue
        # status == 'failed': eligible only if classify already succeeded.
        if not row["path"]:
            continue
        if is_ready_for_retry(row["retry_count"] or 0, row["last_error_at"]):
            pending.append(row["id"])

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
