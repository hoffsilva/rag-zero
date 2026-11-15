"""Loads api/.env.local for local scripts (MCP server, evaluation) that run outside uvicorn."""

import os
from pathlib import Path

ENV_FILE = Path(__file__).parent / ".env.local"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Loads KEY=VALUE lines; variables already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
