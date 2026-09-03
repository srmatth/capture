"""Environment-driven config.

All runtime config comes from env vars set by systemd unit files or the
docker-compose file. Dev override: set them in the shell before running
`uvicorn`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_root: Path
    library_db: Path
    qdrant_url: str
    whisper_bin: Path | None
    whisper_model: Path | None
    hf_home: Path | None
    anthropic_api_key: str | None
    reading_api_url: str

    @classmethod
    def from_env(cls) -> "Config":
        data_root = Path(os.environ.get("DATA_ROOT", "/srv/data/capture"))
        return cls(
            data_root=data_root,
            library_db=Path(os.environ.get("LIBRARY_DB", str(data_root / "library.db"))),
            qdrant_url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
            whisper_bin=_opt_path("WHISPER_BIN"),
            whisper_model=_opt_path("WHISPER_MODEL"),
            hf_home=_opt_path("HF_HOME"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            reading_api_url=os.environ.get("READING_API_URL", "http://localhost:8094/api"),
        )


def _opt_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v) if v else None


# Module-level singleton. Import as `from .config import CONFIG`.
CONFIG = Config.from_env()
