# capture

Personal knowledge capture pipeline for `matthewshome`. Upload photos, voice
memos, and files from the phone; the server transcribes/OCRs, classifies with
Claude, embeds, and makes everything searchable.

Full setup: `../PHASE_2_CAPTURE.md` in the sibling `Projects/` directory.

## Layout

```
app/
  main.py             FastAPI app entrypoint
  db.py               SQLite access + migrations
  config.py           env-driven config
  taxonomy.py         canonical categories + classification prompt
  routers/
    upload.py         POST /upload, /upload_batch, GET /jobs/<id>
    search.py         GET /search, /browse, /item/<id>            (TODO Step 11)
    review.py         GET/POST /inbox                              (TODO Step 12)
  workers/
    transcribe.py     audio + image + pdf → text
    classify.py       text → path + metadata                       (TODO Step 9)
    embed.py          text → vector                                (TODO Step 10)
  migrations/
    001_init.sql      initial schema
  templates/
    index.html        upload PWA
  static/
    upload.js         client-side capture, crop, batch flow
    style.css         minimal styling
    manifest.webmanifest, icons...
```

## Running locally (dev)

Set `DATA_ROOT` to a scratch dir on your dev machine:

```bash
export DATA_ROOT=/tmp/capture-dev
export LIBRARY_DB=/tmp/capture-dev/library.db
export ANTHROPIC_API_KEY=sk-ant-...
mkdir -p /tmp/capture-dev/{inbox/{audio,image,pdf},processed,queue/classify}
uv run uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload
```

Then open http://127.0.0.1:8090 in a browser. Cropper.js assets need to be
vendored under `app/static/` (see PHASE_2_CAPTURE.md Step 6b for the curl
commands).

## Tests

```bash
uv run pytest -x
```

Tests that require Whisper, OCR binaries, or a running Qdrant are marked and
skipped by default; run with `--run-integration` to include them.
