"""Integration tests against a real Postgres with pgvector, without calling OpenAI.

Each test runs in its own temporary schema (dropped afterwards), so it never touches the
chunks table the app uses. Skipped when no database is configured:
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


def database_url() -> str | None:
    from local_env import load_env_file

    load_env_file()
    return os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")


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
    url = database_url()
    if not url:
        pytest.skip("no TEST_DATABASE_URL or DATABASE_URL")
    try:
        conn = await psycopg.AsyncConnection.connect(url, autocommit=True)
    except psycopg.OperationalError as error:
        pytest.skip(f"database not reachable: {error}")

    schema = f"test_{uuid.uuid4().hex[:8]}"
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}", public')  # the vector type lives in public
    await db.ensure_schema(conn)
    try:
        yield conn
    finally:
        await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
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


async def test_switching_model_replaces_the_old_embeddings(connection):
    await db.replace_chunks(connection, [chunk("MON-01-01")], [unit_vector(0)], "old-model")

    await db.replace_chunks(connection, [chunk("MON-01-01")], [unit_vector(0)], MODEL)

    assert await db.count_chunks(connection, "old-model") == 0
    assert await db.count_chunks(connection, MODEL) == 1


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
