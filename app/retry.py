"""Retry policy shared by every worker.

Behavior:

- On worker failure: retry_count += 1, last_error_at = now, status stays
  'failed' unless retry_count crossed MAX_ATTEMPTS, in which case status
  becomes 'dead_letter' (terminal — never retried automatically).
- On worker start: eligible_failed_items() returns 'failed' rows whose
  backoff window has elapsed and whose retry_count is still under
  MAX_ATTEMPTS.
- Manual retry (POST /jobs/<id>/retry): retry_count is reset, status is
  rewound to the pre-failure stage inferred from what fields exist.

Backoff schedule: min(60 * 2^retry_count, 86400) seconds. So the intervals
are 1m, 2m, 4m, 8m, 16m, 32m, 1h4m, 2h8m, 4h16m, then 24h thereafter.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

MAX_ATTEMPTS = 10
BACKOFF_BASE_SECONDS = 60
BACKOFF_CAP_SECONDS = 86400  # 24h


def next_retry_delay(retry_count: int) -> int:
    """Seconds to wait after the retry_count-th failure before attempting
    again. Called with the current retry_count value BEFORE it's
    incremented (so retry_count=0 means 'the first attempt just failed,
    what's the wait before attempt #2?')."""
    wait = BACKOFF_BASE_SECONDS * (2 ** retry_count)
    return min(wait, BACKOFF_CAP_SECONDS)


def is_ready_for_retry(retry_count: int, last_error_at: str | None,
                        now: datetime | None = None) -> bool:
    """True if enough time has passed since the last failure that the
    item is eligible for another attempt. Also True if last_error_at is
    None (never failed before, or was reset by a manual retry).
    """
    if retry_count >= MAX_ATTEMPTS:
        return False
    if not last_error_at:
        return True
    try:
        last = datetime.fromisoformat(last_error_at)
    except ValueError:
        return True  # unparseable → don't hold up the retry
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - last) >= timedelta(seconds=next_retry_delay(retry_count))


def should_dead_letter(retry_count_after_increment: int) -> bool:
    """After bumping retry_count on a fresh failure, has it crossed the
    threshold where we give up? retry_count_after_increment is the value
    the row will have written to the DB, not the pre-failure value."""
    return retry_count_after_increment >= MAX_ATTEMPTS
