-- User-written comments on items. Separate table (not a column) so:
-- - one item can have many comments over time
-- - deleting an item cascades to its comments via FK ON DELETE CASCADE
-- - the item_comments table can be indexed / searched independently
-- - future re-classifications don't clobber the human's notes

CREATE TABLE IF NOT EXISTS item_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_item_comments_item ON item_comments(item_id);
