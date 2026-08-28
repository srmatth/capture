"""Embed worker (Step 10). Not yet implemented — placeholder.

When implemented:
- Read markers from CONFIG.data_root/queue/embed/
- Load item + transcript
- sentence-transformers all-MiniLM-L6-v2 → 384-dim vector
- qdrant_client.upsert into collection 'library' with payload
  {title, path, tags, date_of_content, one_line_summary, uploaded_at}
- Update DB with status='embedded'
- Update items_fts row
- Delete marker
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Step 10 — embed worker not yet built")


if __name__ == "__main__":
    raise SystemExit(main())
