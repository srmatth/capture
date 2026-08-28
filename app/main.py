"""FastAPI entrypoint for the capture app.

One process, two front-ends. The reverse proxy (Caddy) routes both
capture.matthewshome and search.matthewshome here; a Host-header
middleware could later switch between different UIs. For now the same
routes are served on both.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .db import init_db
from .routers import upload

BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="matthewshome capture")

# Initialize DB schema on import so the first request never sees a
# missing table. Idempotent — no-op after the first run.
init_db()

app.include_router(upload.router)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/healthz")
async def healthz() -> dict:
    """Cheap liveness probe. Uptime Kuma hits this."""
    return {"ok": True}
