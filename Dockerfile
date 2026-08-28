FROM python:3.12-slim

# OS-level tools the workers shell out to.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng ocrmypdf ghostscript poppler-utils \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv gives us reproducible builds via uv.lock. Baked in so the container
# can `uv run` its own commands.
RUN pip install --no-cache-dir uv

WORKDIR /srv/capture

# Install deps in their own layer so app-only changes don't invalidate
# the pip cache.
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

COPY app app

# Whisper binary + models get bind-mounted from the host (/srv/data/capture/models/…).
# They live outside the image so upgrading them doesn't force a rebuild.
ENV HF_HOME=/srv/data/capture/models/hf
ENV WHISPER_BIN=/srv/data/capture/models/whisper.cpp/whisper
ENV WHISPER_MODEL=/srv/data/capture/models/whisper.cpp/ggml-medium.en.bin

EXPOSE 8090

CMD ["uv", "run", "--no-dev", \
     "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8090"]
