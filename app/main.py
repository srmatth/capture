"""FastAPI entrypoint for the capture app.

One process, two front-ends. Caddy routes both capture.matthewshome
and search.matthewshome here; a Host-header check on '/' picks the
right landing page. All other routes work on both hosts, so bookmarks
like `search.matthewshome/browse` and `capture.matthewshome/browse`
resolve the same content.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .db import init_db
from .routers import search, upload

BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="matthewshome capture")

# Initialize DB schema on import so the first request never sees a
# missing table. Idempotent — no-op after the first run.
init_db()

app.include_router(upload.router)
app.include_router(search.router)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Root behavior depends on the Host header:
    - search.matthewshome → search landing (mirror of /search)
    - anything else       → capture upload PWA
    """
    host = (request.headers.get("host") or "").lower().split(":")[0]
    if host.startswith("search."):
        # Reuse the search router's landing handler so we don't
        # duplicate template context.
        return await search.search_landing(request, q="")
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/healthz")
async def healthz() -> dict:
    """Cheap liveness probe. Uptime Kuma hits this."""
    return {"ok": True}
