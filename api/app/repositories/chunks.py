"""Chunks and their embeddings in Postgres + pgvector.

Every function takes an open connection and filters on (collection, embedding model): the
caller decides the transaction boundaries.
"""

from collections.abc import Sequence
from typing import Any

from pgvector import Vector

from app.domain import MONDRIA, Chunk, Document, ScoredChunk
from app.repositories.database import Connection

COLUMNS = "id, title, content, document_id, document_title, document_file, source_urls"


async def count_chunks(connection: Connection, model: str, collection: str = MONDRIA) -> int:
    cursor = await connection.execute(
        "SELECT count(*) FROM chunks WHERE collection = %s AND embedding_model = %s", (collection, model)
    )
    row = await cursor.fetchone()
    return int(row[0]) if row else 0  # type: ignore[call-overload]


async def replace_chunks(
    connection: Connection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    model: str,
    collection: str = MONDRIA,
) -> None:
    """Idempotent: upserts every chunk by (collection, section ID, model) and deletes this
    collection's and model's rows for sections that no longer exist, in one transaction.
    Other collections and models stay untouched."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    async with connection.transaction(), connection.cursor() as cursor:
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
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )
        await cursor.execute(
            "DELETE FROM chunks WHERE collection = %s AND embedding_model = %s AND NOT (id = ANY(%s))",
            (collection, model, [chunk["id"] for chunk in chunks]),
        )


def _scored(row: Sequence[Any], score: float) -> ScoredChunk:
    return ScoredChunk(
        id=str(row[0]),
        title=str(row[1]),
        text=str(row[2]),
        document=Document(
            id=str(row[3]), title=str(row[4]), file=str(row[5]), source_urls=[str(url) for url in row[6]]
        ),
        score=score,
        cited=False,
    )


async def search(
    connection: Connection,
    query_embedding: list[float],
    model: str,
    top_k: int,
    collection: str = MONDRIA,
) -> list[ScoredChunk]:
    """Vector search. <=> is cosine distance (0 = same direction), so the closest chunks come
    first with ASC; the score shown to users is the cosine similarity, 1 - distance."""
    return [
        chunk
        for chunk, _ in await search_with_embeddings(connection, query_embedding, model, top_k, collection)
    ]


async def search_with_embeddings(
    connection: Connection,
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
        """,  # noqa: S608 - only the constant COLUMNS is interpolated; values are bound parameters
        {"query": Vector(query_embedding), "model": model, "top_k": top_k, "collection": collection},
    )
    rows: Sequence[Sequence[Any]] = await cursor.fetchall()
    return [(_scored(row, float(row[7])), list(row[8].to_list())) for row in rows]


async def keyword_search(
    connection: Connection,
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
        """,  # noqa: S608 - only the constant COLUMNS is interpolated; values are bound parameters
        {"question": question, "model": model, "top_k": top_k, "collection": collection},
    )
    rows: Sequence[Sequence[Any]] = await cursor.fetchall()
    return [_scored(row, float(row[7])) for row in rows]
