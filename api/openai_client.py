"""Minimal async client for the OpenAI embeddings and chat completions endpoints."""

import os

import httpx

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


async def chat(messages: list[dict[str, str]], json_schema: dict | None = None) -> str:
    body: dict = {"model": chat_model(), "messages": messages}
    if json_schema:
        body["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    result = await _post("/chat/completions", body)
    return (result["choices"][0]["message"]["content"] or "").strip()
