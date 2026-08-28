-- Initial schema. Matches the doc's Step 2 verbatim (PHASE_2_CAPTURE.md).
-- Every uploaded artifact is one row here. Status walks forward through
-- the pipeline; a crash resumes from the last successful stage.

CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,                    -- ULID (26 chars, sortable)
    uploaded_at TEXT NOT NULL,              -- ISO 8601 UTC
    source_kind TEXT NOT NULL,              -- 'audio' | 'image' | 'pdf' | 'plain'
    original_filename TEXT,
    mime_type TEXT,
    size_bytes INTEGER,

    -- Pipeline state. Values, in order:
    --   queued            just uploaded, not yet picked up
    --   transcribing      Whisper / OCR / vision in progress
    --   transcribed       text extracted, awaiting classification
    --   classifying       Haiku call in flight
    --   classified        landed in a destination folder
    --   embedding         sentence-transformers running
    --   embedded          in Qdrant, fully searchable
    --   failed            terminal error; see error_message
    status TEXT NOT NULL,
    error_message TEXT,
    updated_at TEXT NOT NULL,

    upload_note TEXT,                       -- optional user hint at upload time

    -- Post-classification fields (null until 'classified'):
    path TEXT,                              -- e.g. 'notes/personal'
    final_filename TEXT,
    title TEXT,
    one_line_summary TEXT,
    date_of_content TEXT,                   -- ISO 8601 date, may be null
    confidence REAL,                        -- 0..1
    classifier_version TEXT,

    -- Post-transcription (null until 'transcribed'):
    transcript_path TEXT,                   -- relative to DATA_ROOT
    transcript_char_count INTEGER,
    transcript_source TEXT                  -- 'whisper.cpp' | 'tesseract' | 'claude-vision' | 'claude-vision-batch' | 'tesseract-batch' | 'tesseract-pdf' | 'plaintext'
);

CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
CREATE INDEX IF NOT EXISTS idx_items_path ON items(path);
CREATE INDEX IF NOT EXISTS idx_items_uploaded_at ON items(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_items_date_of_content ON items(date_of_content DESC);

CREATE TABLE IF NOT EXISTS item_tags (
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (item_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_item_tags_tag ON item_tags(tag);

CREATE TABLE IF NOT EXISTS item_entities (
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    entity_value TEXT NOT NULL,
    PRIMARY KEY (item_id, entity_type, entity_value)
);

CREATE TABLE IF NOT EXISTS moves (
    -- Audit of every path change for an item. Lets us learn from human
    -- corrections later when the classifier prompt is retrained.
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    moved_at TEXT NOT NULL,
    from_path TEXT,
    to_path TEXT NOT NULL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_moves_item ON moves(item_id);

CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    id UNINDEXED,
    title,
    one_line_summary,
    transcript,
    tokenize = 'porter unicode61 remove_diacritics 1'
);

-- Track applied migrations so db.py knows what's already run.
CREATE TABLE IF NOT EXISTS _migrations (
    name TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);
