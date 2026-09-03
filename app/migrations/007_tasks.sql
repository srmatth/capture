CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    due_at TEXT,
    project TEXT,
    priority TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL DEFAULT 'open',
    source_item_id TEXT REFERENCES items(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    snoozed_until TEXT,
    reminder_token TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_at);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project);
CREATE INDEX IF NOT EXISTS idx_tasks_source ON tasks(source_item_id);
CREATE INDEX IF NOT EXISTS idx_tasks_reminder_token ON tasks(reminder_token);

CREATE TABLE IF NOT EXISTS task_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    UNIQUE(task_id, kind)
);
