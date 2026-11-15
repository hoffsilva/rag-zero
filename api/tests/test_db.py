"""Integration tests against a real Postgres with pgvector, without calling OpenAI.

Each test runs in its own temporary schema (dropped afterwards), so it never touches the
chunks table the app uses. Skipped only when TEST_DATABASE_URL is not set; once it is set,
connection errors fail the run instead of being reported as skips:
    TEST_DATABASE_URL=postgresql://postgres:<password>@localhost:5433/postgres pytest
"""

import os
import uuid

import psycopg
import pytest

import db
from db import Chunk, Document

pytestmark = [pytest.mark.db, pytest.mark.anyio]

DIMENSIONS = 1536
MODEL = "test-model"
DOCUMENT = Document(id="MON-01", title="Entreprise Mondria", file="01.md", source_urls=["https://mondria.com/"])


def unit_vector(axis: int, second_axis: int | None = None) -> list[float]:
    """1 on one axis (optionally 1 on a second one too), normalized to length 1."""
    vector = [0.0] * DIMENSIONS
    vector[axis] = 1.0
    if second_axis is not None:
        vector[second_axis] = 1.0
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector]


def chunk(section_id: str) -> Chunk:
    return Chunk(id=section_id, title=f"Section {section_id}", text=f"## {section_id}\n\ntexte", document=DOCUMENT)


@pytest.fixture
async def connection():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")

    conn = await psycopg.AsyncConnection.connect(url, autocommit=True)
    schema = f"test_{uuid.uuid4().hex[:8]}"
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}", public')  # the vector type lives in public
        await db.ensure_schema(conn)
        yield conn
    finally:
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await conn.close()


async def test_indexing_twice_does_not_duplicate_rows(connection):
    chunks = [chunk("MON-01-01"), chunk("MON-01-02")]
    embeddings = [unit_vector(0), unit_vector(1)]

    await db.replace_chunks(connection, chunks, embeddings, MODEL)
    await db.replace_chunks(connection, chunks, embeddings, MODEL)

    assert await db.count_chunks(connection, MODEL) == 2


async def test_reindexing_removes_sections_deleted_from_the_docs(connection):
    await db.replace_chunks(connection, [chunk("MON-01-01"), chunk("MON-01-02")], [unit_vector(0), unit_vector(1)], MODEL)

    await db.replace_chunks(connection, [chunk("MON-01-01")], [unit_vector(0)], MODEL)

    results = await db.search(connection, unit_vector(1), MODEL, top_k=5)
    assert [r["id"] for r in results] == ["MON-01-01"]


async def test_models_coexist_so_indexing_one_keeps_the_other(connection):
    await db.replace_chunks(connection, [chunk("MON-01-01")], [unit_vector(0)], "small")

    await db.replace_chunks(connection, [chunk("MON-01-01")], [unit_vector(1)], "large")

    assert await db.count_chunks(connection, "small") == 1
    assert await db.count_chunks(connection, "large") == 1
    small = await db.search(connection, unit_vector(0), "small", top_k=1)
    assert small[0]["score"] == pytest.approx(1.0)  # the small vectors were not overwritten


async def test_migrates_a_table_keyed_by_section_id_alone(connection):
    await connection.execute("DROP TABLE chunks")
    await connection.execute(
        """
        CREATE TABLE chunks (
            id text PRIMARY KEY, title text NOT NULL, content text NOT NULL, document_id text NOT NULL,
            document_title text NOT NULL, document_file text NOT NULL, source_urls text[] NOT NULL,
            embedding_model text NOT NULL, embedding vector(1536) NOT NULL
        )
        """
    )

    await db.ensure_schema(connection)
    await db.ensure_schema(connection)  # idempotent

    cursor = await connection.execute(
        """
        SELECT array_agg(a.attname::text ORDER BY a.attname)
        FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
        WHERE i.indrelid = 'chunks'::regclass AND i.indisprimary
        """
    )
    assert (await cursor.fetchone())[0] == ["embedding_model", "id"]


async def test_search_orders_by_cosine_distance(connection):
    await db.replace_chunks(
        connection,
        [chunk("MON-01-01"), chunk("MON-01-02"), chunk("MON-01-03")],
        [unit_vector(0), unit_vector(0, 1), unit_vector(2)],
        MODEL,
    )

    results = await db.search(connection, unit_vector(0), MODEL, top_k=2)

    assert [r["id"] for r in results] == ["MON-01-01", "MON-01-02"]
    assert results[0]["score"] == pytest.approx(1.0)  # same direction: distance 0
    assert results[1]["score"] == pytest.approx(2 ** -0.5)  # 45 degrees: cos = 0.707
    assert results[0]["document"] == DOCUMENT
    assert results[0]["cited"] is False
