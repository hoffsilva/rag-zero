"""Postgres connections and schema."""

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector_async

from app.config import Settings
from app.paths import SCHEMA_FILE

Connection = psycopg.AsyncConnection[tuple[object, ...]]


@dataclass
class Database:
    """Connection settings plus the per-instance "ready" state: the embedding model whose chunks
    this process already checked or indexed. One per app (or script), never module-global."""

    settings: Settings
    ready_model: str | None = None

    async def connect(self) -> Connection:
        return await psycopg.AsyncConnection.connect(self.settings.require_database_url())


async def ensure_schema(connection: Connection) -> None:
    await connection.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
    # The vector type only exists after CREATE EXTENSION, so register it afterwards.
    await register_vector_async(connection)


async def register_vector(connection: Connection) -> None:
    await register_vector_async(connection)
