"""Upload endpoint tests via FastAPI's TestClient. No external services
touched — mime handling, DB rows, and file-on-disk correctness only.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _client():
    from app.main import app
    return TestClient(app)


def _jpeg_bytes() -> bytes:
    """Minimal valid JPEG large enough for img2pdf to derive page size.
    img2pdf refuses images smaller than ~3 PDF units in either dimension."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (400, 400), color=(200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def test_upload_single_image(tmp_data_root: Path) -> None:
    client = _client()
    r = client.post(
        "/upload",
        files={"file": ("scan.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"note": "test note"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    item_id = body["id"]
    assert body["status_url"] == f"/jobs/{item_id}"

    # File landed under inbox/image/<id>.jpg
    landed = list((tmp_data_root / "inbox" / "image").glob(f"{item_id}.*"))
    assert len(landed) == 1 and landed[0].name.endswith(".jpg")

    # DB row exists in status=queued with the note captured.
    from app.db import get_item
    row = get_item(item_id)
    assert row["source_kind"] == "image"
    assert row["upload_note"] == "test note"
    assert row["status"] == "queued"


def test_upload_rejects_unknown_mime(tmp_data_root: Path) -> None:
    client = _client()
    r = client.post(
        "/upload",
        files={"file": ("weird.xyz", b"not-a-known-format", "application/octet-stream")},
    )
    assert r.status_code == 415


def test_upload_batch_stitches_pdf(tmp_data_root: Path) -> None:
    client = _client()
    r = client.post(
        "/upload_batch",
        files=[
            ("files", ("p1.jpg", _jpeg_bytes(), "image/jpeg")),
            ("files", ("p2.jpg", _jpeg_bytes(), "image/jpeg")),
            ("files", ("p3.jpg", _jpeg_bytes(), "image/jpeg")),
        ],
        data={"note": "journal batch"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    item_id = body["id"]
    assert body["pages"] == 3

    # PDF stitched at inbox/pdf/<id>.pdf.
    pdf = tmp_data_root / "inbox" / "pdf" / f"{item_id}.pdf"
    assert pdf.exists() and pdf.stat().st_size > 100

    # Each individual page preserved under inbox/image/<id>/.
    pages = sorted((tmp_data_root / "inbox" / "image" / item_id).glob("page-*.*"))
    assert len(pages) == 3
    assert pages[0].name.startswith("page-01")
    assert pages[-1].name.startswith("page-03")

    # DB row is source_kind='pdf' with the note preserved.
    from app.db import get_item
    row = get_item(item_id)
    assert row["source_kind"] == "pdf"
    assert row["upload_note"] == "journal batch"


def test_upload_batch_rejects_non_image(tmp_data_root: Path) -> None:
    client = _client()
    r = client.post(
        "/upload_batch",
        files=[
            ("files", ("p1.jpg", _jpeg_bytes(), "image/jpeg")),
            ("files", ("bad.txt", b"nope", "text/plain")),
        ],
    )
    assert r.status_code == 415


def test_upload_batch_empty(tmp_data_root: Path) -> None:
    client = _client()
    # FastAPI enforces the required `files` field itself — TestClient
    # sending zero files under that name yields a 422 from Pydantic.
    r = client.post("/upload_batch", files=[])
    assert r.status_code in (400, 422)


def test_job_status_404(tmp_data_root: Path) -> None:
    client = _client()
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404


def test_healthz(tmp_data_root: Path) -> None:
    client = _client()
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
