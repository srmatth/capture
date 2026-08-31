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
from .routers import review, search, taxonomy as taxonomy_router, upload
from .taxonomy import seed_builtins

BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="matthewshome capture")

# Initialize DB schema on import so the first request never sees a
# missing table. Idempotent — no-op after the first run.
init_db()
# Seed the built-in taxonomy after migrations have run. INSERT OR IGNORE
# means user edits to descriptions are preserved across restarts.
seed_builtins()

app.include_router(upload.router)
app.include_router(search.router)
app.include_router(review.router)
app.include_router(taxonomy_router.router)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    """The upload PWA. Reachable at /upload on any host so the nav links
    work whether you're on capture.matthewshome or search.matthewshome."""
    return TEMPLATES.TemplateResponse(request, "index.html")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Root behavior depends on the Host header — this is what the
    home-screen icon and typed-hostname URLs land on:
      - search.matthewshome → search landing (mirror of /search)
      - anything else       → capture upload PWA
    Both landing pages are also reachable at their explicit routes
    (/search, /upload) from the header nav, so the "wrong" subdomain
    is never a dead end — one nav click gets you either way."""
    host = (request.headers.get("host") or "").lower().split(":")[0]
    if host.startswith("search."):
        return await search.search_landing(request, q="")
    return TEMPLATES.TemplateResponse(request, "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    """Cheap liveness probe. Uptime Kuma hits this."""
    return {"ok": True}
