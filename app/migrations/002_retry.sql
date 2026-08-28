-- Retry bookkeeping. See app/retry.py for the policy that reads these.
--
-- retry_count      how many times a worker has failed on this item
-- last_error_at    ISO-8601 UTC timestamp of most recent failure
-- Nothing here removes the existing `error_message` column — that stays
-- as the human-readable text of the latest failure.
--
-- Status values grow one new terminal state: 'dead_letter' — an item that
-- has failed too many times and will no longer be automatically retried.
-- Existing 'failed' remains, but now means "will retry eventually per the
-- backoff schedule."

ALTER TABLE items ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE items ADD COLUMN last_error_at TEXT;
