import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(API_DIR / "eval"))

from app.clients.openai import OpenAIClient  # noqa: E402
from app.config import Settings  # noqa: E402

Handler = Callable[[str, dict[str, Any]], dict[str, Any]]


def openai_transport(handler: Handler) -> httpx.MockTransport:
    """An httpx transport answering OpenAI requests with handler(path, json_body)."""

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        return httpx.Response(200, json=handler(path, json.loads(request.content)))

    return httpx.MockTransport(respond)


def _forbidden(path: str, body: dict[str, Any]) -> dict[str, Any]:
    raise AssertionError(f"unexpected OpenAI call to {path}")


FORBIDDEN_OPENAI = openai_transport(_forbidden)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def no_real_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OpenAIClient created without an explicit transport fails instead of calling OpenAI."""
    original = OpenAIClient.__init__

    def guarded(
        self: OpenAIClient, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        original(self, settings, transport or FORBIDDEN_OPENAI)

    monkeypatch.setattr(OpenAIClient, "__init__", guarded)


@pytest.fixture
def settings() -> Settings:
    """Settings that never read api/.env.local, so tests don't depend on the developer's machine."""
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        openai_chat_model="fake-chat",
        openai_embedding_model="text-embedding-3-small",
        retrieval_strategy="rerank",
        database_url=None,
    )
