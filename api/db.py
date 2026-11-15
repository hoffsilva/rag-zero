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


COLUMNS = "id, title, content, document_id, document_title, document_file, source_urls"


def _scored(row: tuple, score: float) -> ScoredChunk:
    return ScoredChunk(
        id=row[0],
        title=row[1],
        text=row[2],
        document=Document(id=row[3], title=row[4], file=row[5], source_urls=row[6]),
        score=score,
        cited=False,
    )


async def search(
    connection: psycopg.AsyncConnection,
    query_embedding: list[float],
    model: str,
    top_k: int,
) -> list[ScoredChunk]:
    """Vector search. <=> is cosine distance (0 = same direction), so the closest chunks come
    first with ASC; the score shown to users is the cosine similarity, 1 - distance."""
    return [chunk for chunk, _ in await search_with_embeddings(connection, query_embedding, model, top_k)]


async def search_with_embeddings(
    connection: psycopg.AsyncConnection,
    query_embedding: list[float],
    model: str,
    top_k: int,
) -> list[tuple[ScoredChunk, list[float]]]:
    """Vector search that also returns each candidate's embedding (MMR needs them)."""
    cursor = await connection.execute(
        f"""
        SELECT {COLUMNS}, 1 - (embedding <=> %(query)s) AS score, embedding
        FROM chunks
        WHERE embedding_model = %(model)s
        ORDER BY embedding <=> %(query)s
        LIMIT %(top_k)s
        """,
        {"query": Vector(query_embedding), "model": model, "top_k": top_k},
    )
    return [(_scored(row, float(row[7])), row[8].to_list()) for row in await cursor.fetchall()]


async def keyword_search(
    connection: psycopg.AsyncConnection,
    question: str,
    model: str,
    top_k: int,
) -> list[ScoredChunk]:
    """Full-text search with the French configuration (stemming, stop words). The question's
    lexemes are combined with OR, so a chunk doesn't need every word to match; very short
    lexemes ("où", "le") are dropped because they match almost every chunk. ts_rank_cd with
    normalization 1 ranks chunks with more and closer matches higher, damped by chunk length.
    Filtering on the embedding model keeps one row per section when several models coexist."""
    cursor = await connection.execute(
        f"""
        WITH q AS (
            SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS query
            FROM unnest(tsvector_to_array(to_tsvector('french', %(question)s))) AS lexeme
            WHERE length(lexeme) > 2
        )
        SELECT {COLUMNS}, ts_rank_cd(content_tsv, q.query, 1) AS score
        FROM chunks, q
        WHERE embedding_model = %(model)s AND content_tsv @@ q.query
        ORDER BY score DESC, id
        LIMIT %(top_k)s
        """,
        {"question": question, "model": model, "top_k": top_k},
    )
    return [_scored(row, float(row[7])) for row in await cursor.fetchall()]
