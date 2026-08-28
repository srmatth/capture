"""Classify worker (Step 9). Not yet implemented — placeholder so the
package imports cleanly and the systemd unit file can point at it
before Step 9 lands.

When implemented:
- Read markers from CONFIG.data_root/queue/classify/
- Load item + transcript
- Call Haiku with taxonomy.CLASSIFY_PROMPT_TEMPLATE
- Parse JSON, apply CONFIDENCE_FLOOR
- Move raw file inbox/<kind>/<id>.<ext> → <path>/<id>.<ext>
- Write .meta.json under processed/<path>/
- Update DB with status='classified' + all fields
- Drop marker in queue/embed/<id>
- Delete the classify marker
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Step 9 — classify worker not yet built")


if __name__ == "__main__":
    raise SystemExit(main())
