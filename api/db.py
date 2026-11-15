"""Postgres + pgvector storage for the chunks and their embeddings."""

import os
from pathlib import Path
from typing import TypedDict

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector_async

SCHEMA_FILE = Path(__file__).parent / "schema.sql"


class Document(TypedDict):
    id: str
    title: str
    file: str
    source_urls: list[str]


class Chunk(TypedDict):
    id: str
    title: str
    text: str
    document: Document


class ScoredChunk(Chunk):
    score: float
    cited: bool


async def connect() -> psycopg.AsyncConnection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    connection = await psycopg.AsyncConnection.connect(url)
    return connection


async def ensure_schema(connection: psycopg.AsyncConnection) -> None:
    await connection.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
    # The vector type only exists after CREATE EXTENSION, so register it afterwards.
    await register_vector_async(connection)


async def count_chunks(connection: psycopg.AsyncConnection, model: str) -> int:
    cursor = await connection.execute("SELECT count(*) FROM chunks WHERE embedding_model = %s", (model,))
    row = await cursor.fetchone()
    return row[0] if row else 0


async def replace_chunks(
    connection: psycopg.AsyncConnection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    model: str,
) -> None:
    """Idempotent: upserts every chunk by (section ID, model) and deletes this model's rows for
    sections that no longer exist in the docs, in one transaction. Other models' rows stay."""
    async with connection.transaction():
        async with connection.cursor() as cursor:
            await cursor.executemany(
                """
                INSERT INTO chunks (id, title, content, document_id, document_title, document_file,
                                    source_urls, embedding_model, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id, embedding_model) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    document_id = EXCLUDED.document_id,
                    document_title = EXCLUDED.document_title,
                    document_file = EXCLUDED.document_file,
                    source_urls = EXCLUDED.source_urls,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        chunk["id"],
                        chunk["title"],
                        chunk["text"],
                        chunk["document"]["id"],
                        chunk["document"]["title"],
                        chunk["document"]["file"],
                        chunk["document"]["source_urls"],
                        model,
                        Vector(embedding),
                    )
                    for chunk, embedding in zip(chunks, embeddings)
                ],
            )
            await cursor.execute(
                "DELETE FROM chunks WHERE embedding_model = %s AND NOT (id = ANY(%s))",
                (model, [chunk["id"] for chunk in chunks]),
            )


async def search(
    connection: psycopg.AsyncConnection,
    query_embedding: list[float],
    model: str,
    top_k: int,
) -> list[ScoredChunk]:
    # <=> is cosine distance (0 = same direction), so the closest chunks come first with ASC.
    # The score shown to users is the cosine similarity, 1 - distance.
    cursor = await connection.execute(
        """
        SELECT id, title, content, document_id, document_title, document_file, source_urls,
               1 - (embedding <=> %(query)s) AS score
        FROM chunks
        WHERE embedding_model = %(model)s
        ORDER BY embedding <=> %(query)s
        LIMIT %(top_k)s
        """,
        {"query": Vector(query_embedding), "model": model, "top_k": top_k},
    )
    return [
        ScoredChunk(
            id=row[0],
            title=row[1],
            text=row[2],
            document=Document(id=row[3], title=row[4], file=row[5], source_urls=row[6]),
            score=float(row[7]),
            cited=False,
        )
        for row in await cursor.fetchall()
    ]
