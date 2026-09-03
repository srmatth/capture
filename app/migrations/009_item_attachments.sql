CREATE TABLE IF NOT EXISTS item_attachments (
    id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    mime_type TEXT,
    size_bytes INTEGER,
    uploaded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attachments_item ON item_attachments(item_id);
