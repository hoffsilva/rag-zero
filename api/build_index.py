"""Indexes mondria-rag-docs/ into Postgres (table chunks) with the configured embedding model.

Idempotent: running it twice updates the same rows instead of duplicating them.
The API also indexes on its first request when the table has no chunks for the model.

Run from this folder with the variables in the environment:
    set -a && source .env.local && set +a && python build_index.py
"""

import asyncio

import db
from openai_client import embedding_model
from rag import index_documents


async def main() -> None:
    model = embedding_model()
    async with await db.connect() as connection:
        await db.ensure_schema(connection)
        count = await index_documents(connection, model)
        stored = await db.count_chunks(connection, model)
    print(f"Indexed {count} chunks with {model}; table chunks now has {stored} rows for this model")


if __name__ == "__main__":
    asyncio.run(main())
