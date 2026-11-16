"""Command-line entry points (see the wrappers in api/)."""

import asyncio

from app.repositories import chunks as chunk_repository
from app.repositories.database import ensure_schema
from app.runtime import open_runtime
from app.services.indexing import index_documents


async def _index() -> None:
    async with open_runtime() as runtime, await runtime.database.connect() as connection:
        model = runtime.openai.embedding_model
        await ensure_schema(connection)
        count = await index_documents(runtime.openai, connection, model)
        stored = await chunk_repository.count_chunks(connection, model)
    print(f"Indexed {count} chunks with {model}; table chunks now has {stored} rows for this model")


def index() -> None:
    """Indexes mondria-rag-docs/ into Postgres with the configured embedding model (idempotent)."""
    asyncio.run(_index())
