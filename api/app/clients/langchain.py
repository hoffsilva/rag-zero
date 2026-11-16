"""LangChain chat and embedding models, used by the LangChain pipeline (services/langchain_rag.py).

They share the OpenAIClient's httpx connection pool, so they live and close with it, and tests
that inject a fake transport into OpenAIClient cover LangChain too. The pipeline is async end to
end; the sync client refuses every request so a blocking call can't slip in unnoticed.
"""

from functools import cached_property

import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from app.clients.openai import (
    EMBEDDING_DIMENSIONS,
    MAX_ANSWER_TOKENS,
    OPENAI_BASE_URL,
    TIMEOUT_SECONDS,
    OpenAIClient,
)
from app.config import Settings


class _AsyncOnly(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise RuntimeError("The LangChain models are used asynchronously only")


class LangChainModels:
    """Built on first use (keyword arguments use the field names: mypy's pydantic plugin
    doesn't accept the aliases), so a missing key or chat model only fails the requests that need them."""

    def __init__(self, settings: Settings, openai: OpenAIClient) -> None:
        self._settings = settings
        self._openai = openai

    @cached_property
    def chat(self) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=self._settings.require_chat_model(),
            openai_api_key=SecretStr(self._settings.require_openai_api_key()),
            openai_api_base=OPENAI_BASE_URL,
            max_tokens=MAX_ANSWER_TOKENS,  # sent as max_completion_tokens
            request_timeout=TIMEOUT_SECONDS,
            http_async_client=self._openai.http,
            http_client=httpx.Client(transport=_AsyncOnly()),
        )

    @cached_property
    def embeddings(self) -> OpenAIEmbeddings:
        model = self._settings.openai_embedding_model
        return OpenAIEmbeddings(
            model=model,
            # Same rule as OpenAIClient.embed: text-embedding-3 models are asked for the column size.
            dimensions=EMBEDDING_DIMENSIONS if model.startswith("text-embedding-3") else None,
            # Skips the tiktoken length check, which downloads a tokenizer file at run time.
            check_embedding_ctx_length=False,
            openai_api_key=SecretStr(self._settings.require_openai_api_key()),
            openai_api_base=OPENAI_BASE_URL,
            request_timeout=TIMEOUT_SECONDS,
            http_async_client=self._openai.http,
            http_client=httpx.Client(transport=_AsyncOnly()),
        )
