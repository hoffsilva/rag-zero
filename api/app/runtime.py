"""What every entry point (HTTP app, MCP stdio server, indexer, evaluation) needs at run time."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from app.clients.langchain import LangChainModels
from app.clients.openai import OpenAIClient
from app.config import Settings, get_settings
from app.repositories.database import Database


@dataclass(frozen=True)
class Runtime:
    settings: Settings
    openai: OpenAIClient
    database: Database
    langchain: LangChainModels


@asynccontextmanager
async def open_runtime(
    settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> AsyncIterator[Runtime]:
    """Opens the OpenAI client (closed on exit, also on errors) and the database handle."""
    settings = settings or get_settings()
    async with OpenAIClient(settings, transport) as openai:
        yield Runtime(
            settings=settings,
            openai=openai,
            database=Database(settings),
            langchain=LangChainModels(settings, openai),
        )
