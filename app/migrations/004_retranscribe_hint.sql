-- Async retranscribe support. When the /item/<id>/retranscribe endpoint
-- fires, we set retranscribe_hint to 'vision' / 'tesseract' / 'force-ocr'
-- and reset status to 'queued'. The transcribe worker checks the hint on
-- pickup; if set, uses that method regardless of the usual routing, then
-- clears the column back to NULL. This makes retranscribe follow the same
-- async pattern as uploads — the endpoint returns immediately with a job
-- ID, and the workers process out-of-band.
ALTER TABLE items ADD COLUMN retranscribe_hint TEXT;
