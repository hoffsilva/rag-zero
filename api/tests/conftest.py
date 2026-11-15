import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(API_DIR / "eval"))

# run_eval and mcp_server load api/.env.local when imported. Disable that before any test module
# imports them, so the suite never picks up the real OpenAI key or the app's database.
import local_env  # noqa: E402

local_env.load_env_file = lambda *args, **kwargs: None

import openai_client  # noqa: E402


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def no_openai_calls(monkeypatch):
    """Any test that reaches the real OpenAI client fails instead of making a network call."""

    async def forbidden(path: str, body: dict) -> dict:
        raise AssertionError(f"unexpected OpenAI call to {path}")

    monkeypatch.setattr(openai_client, "_post", forbidden)
