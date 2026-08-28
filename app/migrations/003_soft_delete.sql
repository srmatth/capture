-- Soft delete. deleted_at NULL = live; non-NULL = tombstoned.
-- All search / browse / detail routes filter WHERE deleted_at IS NULL by
-- default. A future Trash view can list deleted rows explicitly.
ALTER TABLE items ADD COLUMN deleted_at TEXT;
CREATE INDEX IF NOT EXISTS idx_items_deleted ON items(deleted_at);
