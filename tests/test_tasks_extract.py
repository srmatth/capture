"""Tests for the task extraction module."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _make_response(tasks_list: list[dict]) -> MagicMock:
    resp = MagicMock()
    block = MagicMock()
    block.text = json.dumps({"tasks": tasks_list})
    resp.content = [block]
    return resp


@patch("app.tasks_extract.anthropic.Anthropic")
def test_extract_happy_path(mock_cls, tmp_data_root):
    from app.tasks_extract import extract_tasks

    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_response([
        {"title": "Call Mom", "due_at": "2026-09-10T14:00:00", "project": None, "priority": "normal"},
        {"title": "Buy groceries", "due_at": None, "project": "house", "priority": "low"},
    ])

    result = extract_tasks("I need to call mom on September 10th and buy groceries", "ITEM1")
    assert len(result) == 2
    assert result[0]["title"] == "Call Mom"
    assert result[1]["project"] == "house"


@patch("app.tasks_extract.anthropic.Anthropic")
def test_extract_refuses_more_than_five(mock_cls, tmp_data_root):
    from app.tasks_extract import extract_tasks

    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_response([
        {"title": f"Task {i}", "due_at": None, "project": None, "priority": "normal"}
        for i in range(7)
    ])

    result = extract_tasks("lots of tasks mentioned here in a long transcript", "ITEM2")
    assert result == []


@patch("app.tasks_extract.anthropic.Anthropic")
def test_extract_drops_past_dates(mock_cls, tmp_data_root):
    from app.tasks_extract import extract_tasks

    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_response([
        {"title": "Old task", "due_at": "2020-01-01T00:00:00", "priority": "normal"},
    ])

    result = extract_tasks("do the old thing from long ago in a voice memo", "ITEM3")
    assert len(result) == 1
    assert result[0]["due_at"] is None


def test_extract_empty_transcript(tmp_data_root):
    from app.tasks_extract import extract_tasks
    assert extract_tasks("", "ITEM4") == []


def test_extract_short_transcript(tmp_data_root):
    from app.tasks_extract import extract_tasks
    assert extract_tasks("hi", "ITEM5") == []


@patch("app.tasks_extract.anthropic.Anthropic")
def test_extract_invalid_json(mock_cls, tmp_data_root):
    from app.tasks_extract import extract_tasks

    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    resp = MagicMock()
    block = MagicMock()
    block.text = "not valid json at all"
    resp.content = [block]
    mock_client.messages.create.return_value = resp

    result = extract_tasks("some transcript here that is long enough to pass", "ITEM6")
    assert result == []


@patch("app.tasks_extract.anthropic.Anthropic")
def test_extract_validates_priority(mock_cls, tmp_data_root):
    from app.tasks_extract import extract_tasks

    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_response([
        {"title": "Weird priority", "due_at": None, "priority": "URGENT"},
    ])

    result = extract_tasks("do something with weird priority value mentioned here", "ITEM7")
    assert len(result) == 1
    assert result[0]["priority"] == "normal"
