"""Tests for task DB functions."""
from __future__ import annotations


def _init():
    from app.db import init_db
    from app.taxonomy import seed_builtins
    init_db()
    seed_builtins()


def test_insert_and_get_task(tmp_data_root):
    _init()
    from app.db import get_task, insert_task

    insert_task(
        task_id="TASK001",
        title="Call Mom",
        due_at="2026-09-10T14:00:00+00:00",
        project=None,
        priority="normal",
        source_item_id=None,
        reminder_token="tok001",
    )
    task = get_task("TASK001")
    assert task is not None
    assert task["title"] == "Call Mom"
    assert task["priority"] == "normal"
    assert task["status"] == "open"
    assert task["reminder_token"] == "tok001"


def test_update_task(tmp_data_root):
    _init()
    from app.db import get_task, insert_task, update_task

    insert_task(task_id="TASK002", title="Test", reminder_token="tok002")
    update_task("TASK002", status="done", completed_at="2026-09-10")
    task = get_task("TASK002")
    assert task["status"] == "done"


def test_delete_task(tmp_data_root):
    _init()
    from app.db import delete_task, get_task, insert_task

    insert_task(task_id="TASK003", title="Delete me", reminder_token="tok003")
    delete_task("TASK003")
    assert get_task("TASK003") is None


def test_list_open_tasks_ordering(tmp_data_root):
    _init()
    from app.db import insert_task, list_open_tasks

    insert_task(task_id="T1", title="Low", priority="low", reminder_token="t1")
    insert_task(task_id="T2", title="High", priority="high",
                due_at="2026-09-15", reminder_token="t2")
    insert_task(task_id="T3", title="Normal", priority="normal",
                due_at="2026-09-10", reminder_token="t3")

    tasks = list_open_tasks()
    assert len(tasks) == 3
    assert tasks[0]["title"] == "High"
    assert tasks[1]["title"] == "Normal"
    assert tasks[2]["title"] == "Low"


def test_is_duplicate_exact(tmp_data_root):
    _init()
    from app.db import insert_task, is_duplicate_task

    insert_task(task_id="DUP1", title="Buy milk", due_at="2026-09-10",
                reminder_token="d1")
    assert is_duplicate_task("Buy milk", "2026-09-10") is True
    assert is_duplicate_task("Buy milk", "2026-09-11") is False


def test_is_duplicate_case_insensitive(tmp_data_root):
    _init()
    from app.db import insert_task, is_duplicate_task

    insert_task(task_id="DUP2", title="Buy Milk", reminder_token="d2")
    assert is_duplicate_task("buy milk", None) is True


def test_is_duplicate_levenshtein(tmp_data_root):
    _init()
    from app.db import insert_task, is_duplicate_task

    insert_task(task_id="DUP3", title="Buy milk", reminder_token="d3")
    assert is_duplicate_task("Buy milks", None) is True


def test_is_duplicate_prefix(tmp_data_root):
    _init()
    from app.db import insert_task, is_duplicate_task

    insert_task(task_id="DUP4", title="Buy groceries", reminder_token="d4")
    assert is_duplicate_task("Buy groceries from Trader Joe's", None) is True


def test_is_duplicate_different_dates_not_dup(tmp_data_root):
    _init()
    from app.db import insert_task, is_duplicate_task

    insert_task(task_id="DUP5", title="Call Mom", due_at="2026-09-10",
                reminder_token="d5")
    assert is_duplicate_task("Call Mom", "2026-09-15") is False


def test_generate_token_unique(tmp_data_root):
    from app.db import generate_reminder_token

    t1 = generate_reminder_token()
    t2 = generate_reminder_token()
    assert t1 != t2
    assert len(t1) > 10


def test_task_alert_idempotent(tmp_data_root):
    _init()
    from app.db import get_task_alerts, insert_task, insert_task_alert

    insert_task(task_id="ALERT1", title="Test", reminder_token="a1")
    assert insert_task_alert("ALERT1", "7d") is True
    assert insert_task_alert("ALERT1", "7d") is False

    alerts = get_task_alerts("ALERT1")
    assert len(alerts) == 1
    assert alerts[0]["kind"] == "7d"


def test_get_task_by_reminder_token(tmp_data_root):
    _init()
    from app.db import get_task_by_reminder_token, insert_task

    insert_task(task_id="TOK1", title="Token lookup", reminder_token="unique_tok")
    found = get_task_by_reminder_token("unique_tok")
    assert found is not None
    assert found["id"] == "TOK1"

    assert get_task_by_reminder_token("nonexistent") is None


def test_levenshtein():
    from app.db import _levenshtein

    assert _levenshtein("kitten", "sitting") == 3
    assert _levenshtein("", "abc") == 3
    assert _levenshtein("abc", "abc") == 0
    assert _levenshtein("Buy milk", "Buy milks") == 1
