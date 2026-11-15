"""Postgres + pgvector storage for the chunks and their embeddings."""

import os
from pathlib import Path
from typing import TypedDict

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector_async

SCHEMA_FILE = Path(__file__).parent / "schema.sql"

MONDRIA = "mondria"  # collection of the demo knowledge base
UPLOAD_TTL_HOURS = 24


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


async def count_chunks(connection: psycopg.AsyncConnection, model: str, collection: str = MONDRIA) -> int:
    cursor = await connection.execute(
        "SELECT count(*) FROM chunks WHERE collection = %s AND embedding_model = %s", (collection, model)
    )
    row = await cursor.fetchone()
    return row[0] if row else 0


async def replace_chunks(
    connection: psycopg.AsyncConnection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    model: str,
    collection: str = MONDRIA,
) -> None:
    """Idempotent: upserts every chunk by (collection, section ID, model) and deletes this
    collection's and model's rows for sections that no longer exist, in one transaction.
    Other collections and models stay untouched."""
    async with connection.transaction():
        async with connection.cursor() as cursor:
            await cursor.executemany(
                """
                INSERT INTO chunks (collection, id, title, content, document_id, document_title,
                                    document_file, source_urls, embedding_model, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (collection, id, embedding_model) DO UPDATE SET
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
                        collection,
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
                "DELETE FROM chunks WHERE collection = %s AND embedding_model = %s AND NOT (id = ANY(%s))",
                (collection, model, [chunk["id"] for chunk in chunks]),
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
    collection: str = MONDRIA,
) -> list[ScoredChunk]:
    """Vector search. <=> is cosine distance (0 = same direction), so the closest chunks come
    first with ASC; the score shown to users is the cosine similarity, 1 - distance."""
    return [chunk for chunk, _ in await search_with_embeddings(connection, query_embedding, model, top_k, collection)]


async def search_with_embeddings(
    connection: psycopg.AsyncConnection,
    query_embedding: list[float],
    model: str,
    top_k: int,
    collection: str = MONDRIA,
) -> list[tuple[ScoredChunk, list[float]]]:
    """Vector search that also returns each candidate's embedding (MMR needs them)."""
    cursor = await connection.execute(
        f"""
        SELECT {COLUMNS}, 1 - (embedding <=> %(query)s) AS score, embedding
        FROM chunks
        WHERE collection = %(collection)s AND embedding_model = %(model)s
        ORDER BY embedding <=> %(query)s
        LIMIT %(top_k)s
        """,
        {"query": Vector(query_embedding), "model": model, "top_k": top_k, "collection": collection},
    )
    return [(_scored(row, float(row[7])), row[8].to_list()) for row in await cursor.fetchall()]


async def keyword_search(
    connection: psycopg.AsyncConnection,
    question: str,
    model: str,
    top_k: int,
    collection: str = MONDRIA,
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
        WHERE collection = %(collection)s AND embedding_model = %(model)s AND content_tsv @@ q.query
        ORDER BY score DESC, id
        LIMIT %(top_k)s
        """,
        {"question": question, "model": model, "top_k": top_k, "collection": collection},
    )
    return [_scored(row, float(row[7])) for row in await cursor.fetchall()]


class Upload(TypedDict):
    collection: str
    title: str
    file: str
    kind: str
    size_bytes: int
    sections: list[str]
    language: str
    summary: str
    suggested_questions: list[str]
    expires_at: str


async def save_upload(connection: psycopg.AsyncConnection, upload: Upload) -> None:
    await connection.execute(
        """
        INSERT INTO uploads (collection, title, file, kind, size_bytes, sections, language, summary,
                             suggested_questions)
        VALUES (%(collection)s, %(title)s, %(file)s, %(kind)s, %(size_bytes)s, %(sections)s, %(language)s,
                %(summary)s, %(suggested_questions)s)
        """,
        dict(upload),
    )


async def get_upload(connection: psycopg.AsyncConnection, collection: str) -> Upload | None:
    """The upload's metadata, or None when it doesn't exist or has expired."""
    cursor = await connection.execute(
        f"""
        SELECT collection, title, file, kind, size_bytes, sections, language, summary, suggested_questions,
               created_at + interval '{UPLOAD_TTL_HOURS} hours'
        FROM uploads
        WHERE collection = %s AND created_at > now() - interval '{UPLOAD_TTL_HOURS} hours'
        """,
        (collection,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return Upload(
        collection=row[0], title=row[1], file=row[2], kind=row[3], size_bytes=row[4], sections=row[5],
        language=row[6], summary=row[7], suggested_questions=row[8], expires_at=row[9].isoformat(),
    )


async def delete_expired_uploads(connection: psycopg.AsyncConnection) -> None:
    """Uploads live 24 h. Called on every new upload, so no scheduled job is needed."""
    async with connection.transaction():
        await connection.execute(
            f"DELETE FROM chunks WHERE collection LIKE 'upload-%%' AND created_at < now() - interval '{UPLOAD_TTL_HOURS} hours'"
        )
        await connection.execute(
            f"DELETE FROM uploads WHERE created_at < now() - interval '{UPLOAD_TTL_HOURS} hours'"
        )
