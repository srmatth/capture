-- Migration 006: move the classifier taxonomy into a table so it's
-- editable from the UI. The classify worker and the /taxonomy admin
-- page both read from here; the seed of built-ins happens in
-- app/taxonomy.py at startup, right after init_db() runs.
--
-- is_builtin=1 protects an entry from deletion. Descriptions on
-- built-ins are still editable (they tune the classifier prompt).

CREATE TABLE IF NOT EXISTS taxonomy_entries (
    path TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    is_builtin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
