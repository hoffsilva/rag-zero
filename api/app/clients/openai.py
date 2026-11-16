"""Async client for the OpenAI embeddings and chat completions endpoints.

One instance owns one httpx.AsyncClient: open it with `async with OpenAIClient(...)`, which
closes the connection pool even on errors or cancellation. The HTTP app opens one in its
lifespan; the MCP stdio server, the indexer and the evaluation each open their own.
"""

import logging
from types import TracebackType
from typing import Any, Self, cast

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

OPENAI_BASE_URL = "https://api.openai.com/v1"
TIMEOUT_SECONDS = 30
EMBEDDING_DIMENSIONS = 1536  # must match vector(1536) in schema.sql
# Hard cap on the generated answer: bounds the cost of every public request, whatever the prompt
# says. A concise answer with citations is well under 200 tokens.
MAX_ANSWER_TOKENS = 500
CHAT_ATTEMPTS = 3  # retries for corrupted output (see chat)

JsonObject = dict[str, Any]


class ChatRefusal(Exception):  # noqa: N818 - it is a model outcome, not an error of this code
    """The model declined to answer (Structured Outputs returns a refusal instead of JSON)."""


class OpenAIError(RuntimeError):
    """OpenAI answered with an HTTP error."""


class OpenAIClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport  # tests inject httpx.MockTransport here
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        """The open connection pool, shared with the LangChain models (see clients/langchain.py)."""
        if self._http is None:
            raise RuntimeError("Open the client with `async with OpenAIClient(...)` first")
        return self._http

    @property
    def embedding_model(self) -> str:
        return self._settings.openai_embedding_model

    @property
    def chat_model(self) -> str:
        return self._settings.require_chat_model()

    async def __aenter__(self) -> Self:
        self._http = httpx.AsyncClient(
            base_url=OPENAI_BASE_URL, timeout=TIMEOUT_SECONDS, transport=self._transport
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _post(self, path: str, body: JsonObject) -> JsonObject:
        if self._http is None:
            raise RuntimeError("OpenAIClient used outside of `async with`")
        response = await self._http.post(
            path,
            json=body,
            headers={"Authorization": f"Bearer {self._settings.require_openai_api_key()}"},
        )
        if response.is_error:
            raise OpenAIError(f"OpenAI {path} failed ({response.status_code}): {response.text}")
        return cast(JsonObject, response.json())

    async def embed(self, inputs: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.embedding_model
        body: JsonObject = {"model": model, "input": inputs}
        if model.startswith("text-embedding-3"):
            # text-embedding-3-large returns 3072 dimensions by default; ask for the column size.
            body["dimensions"] = EMBEDDING_DIMENSIONS
        result = await self._post("/embeddings", body)
        items = sorted(result["data"], key=lambda item: item["index"])
        return [[float(value) for value in item["embedding"]] for item in items]

    async def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: JsonObject | None = None,
        json_mode: bool = False,
    ) -> str:
        """json_schema: strict Structured Outputs. json_mode: valid JSON without a strict schema."""
        body: JsonObject = {
            "model": self.chat_model,
            "messages": messages,
            "max_completion_tokens": MAX_ANSWER_TOKENS,
        }
        if json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        elif json_mode:
            body["response_format"] = {"type": "json_object"}

        content = ""
        for attempt in range(1, CHAT_ATTEMPTS + 1):
            result = await self._post("/chat/completions", body)
            choice = result["choices"][0]
            message = choice["message"]
            if message.get("refusal"):
                raise ChatRefusal(message["refusal"])
            if choice.get("finish_reason") != "stop":
                # "length" means the answer hit MAX_ANSWER_TOKENS; truncated JSON can't be parsed.
                raise OpenAIError(
                    f"Chat completion did not finish (finish_reason={choice.get('finish_reason')})"
                )
            content = str(message.get("content") or "").strip()
            # Seen in production: structured output where accented letters came back as "\u0000"
            # ("employés" -> "employ\u0000s"), intermittently. Postgres rejects NUL characters and
            # the text is unreadable, so ask again; the last attempt is returned with them removed.
            if "\x00" not in content:
                return content
            logger.warning("Chat output contained NUL characters (attempt %d of %d)", attempt, CHAT_ATTEMPTS)
        return content.replace("\x00", "")
