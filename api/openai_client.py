"""Minimal async client for the OpenAI embeddings and chat completions endpoints."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def embedding_model() -> str:
    return os.environ.get("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def chat_model() -> str:
    return _env("OPENAI_CHAT_MODEL")


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


async def _post(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(base_url=OPENAI_BASE_URL, timeout=30) as client:
        response = await client.post(
            path,
            json=body,
            headers={"Authorization": f"Bearer {_env('OPENAI_API_KEY')}"},
        )
    if response.is_error:
        raise RuntimeError(f"OpenAI {path} failed ({response.status_code}): {response.text}")
    return response.json()


EMBEDDING_DIMENSIONS = 1536  # must match vector(1536) in schema.sql


async def embed(inputs: list[str], model: str) -> list[list[float]]:
    body: dict = {"model": model, "input": inputs}
    if model.startswith("text-embedding-3"):
        # text-embedding-3-large returns 3072 dimensions by default; ask for the column size.
        body["dimensions"] = EMBEDDING_DIMENSIONS
    result = await _post("/embeddings", body)
    return [item["embedding"] for item in sorted(result["data"], key=lambda item: item["index"])]


# Hard cap on the generated answer: bounds the cost of every public request, whatever the prompt
# says. A concise answer with citations is well under 200 tokens.
MAX_ANSWER_TOKENS = 500


class ChatRefusal(Exception):
    """The model declined to answer (Structured Outputs returns a refusal instead of JSON)."""


CHAT_ATTEMPTS = 3  # retries for corrupted output (see below)


async def chat(
    messages: list[dict[str, str]], json_schema: dict | None = None, json_mode: bool = False
) -> str:
    """json_schema: strict Structured Outputs. json_mode: valid JSON without a strict schema."""
    body: dict = {"model": chat_model(), "messages": messages, "max_completion_tokens": MAX_ANSWER_TOKENS}
    if json_schema:
        body["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    elif json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(1, CHAT_ATTEMPTS + 1):
        result = await _post("/chat/completions", body)
        choice = result["choices"][0]
        message = choice["message"]
        if message.get("refusal"):
            raise ChatRefusal(message["refusal"])
        if choice.get("finish_reason") != "stop":
            # "length" means the answer hit MAX_ANSWER_TOKENS; truncated JSON can't be parsed.
            raise RuntimeError(f"Chat completion did not finish (finish_reason={choice.get('finish_reason')})")
        content = (message.get("content") or "").strip()
        # Seen in production: structured output where accented letters came back as "\u0000"
        # ("employés" -> "employ\u0000s"), intermittently. Postgres rejects NUL characters and the
        # text is unreadable, so ask again; the last attempt is returned with them removed.
        if "\x00" not in content:
            return content
        logger.warning("Chat output contained NUL characters (attempt %d of %d)", attempt, CHAT_ATTEMPTS)
    return content.replace("\x00", "")
