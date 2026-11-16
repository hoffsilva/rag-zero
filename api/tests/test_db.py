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

from app.domain import Chunk, Document, Upload
from app.repositories import chunks as chunk_repository
from app.repositories import uploads as upload_repository
from app.repositories.database import ensure_schema

pytestmark = [pytest.mark.db, pytest.mark.anyio]

DIMENSIONS = 1536
MODEL = "test-model"
DOCUMENT = Document(
    id="MON-01", title="Entreprise Mondria", file="01.md", source_urls=["https://mondria.com/"]
)


def unit_vector(axis: int, second_axis: int | None = None) -> list[float]:
    """1 on one axis (optionally 1 on a second one too), normalized to length 1."""
    vector = [0.0] * DIMENSIONS
    vector[axis] = 1.0
    if second_axis is not None:
        vector[second_axis] = 1.0
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector]


def chunk(section_id: str) -> Chunk:
    return Chunk(
        id=section_id, title=f"Section {section_id}", text=f"## {section_id}\n\ntexte", document=DOCUMENT
    )


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
        await ensure_schema(conn)
        yield conn
    finally:
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await conn.close()


async def test_indexing_twice_does_not_duplicate_rows(connection):
    chunks = [chunk("MON-01-01"), chunk("MON-01-02")]
    embeddings = [unit_vector(0), unit_vector(1)]

    await chunk_repository.replace_chunks(connection, chunks, embeddings, MODEL)
    await chunk_repository.replace_chunks(connection, chunks, embeddings, MODEL)

    assert await chunk_repository.count_chunks(connection, MODEL) == 2


async def test_reindexing_removes_sections_deleted_from_the_docs(connection):
    await chunk_repository.replace_chunks(
        connection, [chunk("MON-01-01"), chunk("MON-01-02")], [unit_vector(0), unit_vector(1)], MODEL
    )

    await chunk_repository.replace_chunks(connection, [chunk("MON-01-01")], [unit_vector(0)], MODEL)

    results = await chunk_repository.search(connection, unit_vector(1), MODEL, top_k=5)
    assert [r["id"] for r in results] == ["MON-01-01"]


async def test_models_coexist_so_indexing_one_keeps_the_other(connection):
    await chunk_repository.replace_chunks(connection, [chunk("MON-01-01")], [unit_vector(0)], "small")

    await chunk_repository.replace_chunks(connection, [chunk("MON-01-01")], [unit_vector(1)], "large")

    assert await chunk_repository.count_chunks(connection, "small") == 1
    assert await chunk_repository.count_chunks(connection, "large") == 1
    small = await chunk_repository.search(connection, unit_vector(0), "small", top_k=1)
    assert small[0]["score"] == pytest.approx(1.0)  # the small vectors were not overwritten


async def test_migrates_a_table_created_before_collections(connection):
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

    await ensure_schema(connection)
    await ensure_schema(connection)  # idempotent

    cursor = await connection.execute(
        """
        SELECT array_agg(a.attname::text ORDER BY a.attname)
        FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
        WHERE i.indrelid = 'chunks'::regclass AND i.indisprimary
        """
    )
    assert (await cursor.fetchone())[0] == ["collection", "embedding_model", "id"]


async def test_search_orders_by_cosine_distance(connection):
    await chunk_repository.replace_chunks(
        connection,
        [chunk("MON-01-01"), chunk("MON-01-02"), chunk("MON-01-03")],
        [unit_vector(0), unit_vector(0, 1), unit_vector(2)],
        MODEL,
    )

    results = await chunk_repository.search(connection, unit_vector(0), MODEL, top_k=2)

    assert [r["id"] for r in results] == ["MON-01-01", "MON-01-02"]
    assert results[0]["score"] == pytest.approx(1.0)  # same direction: distance 0
    assert results[1]["score"] == pytest.approx(2**-0.5)  # 45 degrees: cos = 0.707
    assert results[0]["document"] == DOCUMENT
    assert results[0]["cited"] is False


async def test_keyword_search_matches_french_word_forms(connection):
    office = Chunk(
        id="MON-01-04",
        title="Coordonnées",
        text="Bureaux publiés à Sherbrooke et Boisbriand.",
        document=DOCUMENT,
    )
    nonprofits = Chunk(
        id="MON-05-01",
        title="Secteurs visés",
        text="Organismes sans but lucratif (OBNL) et commerce de détail.",
        document=DOCUMENT,
    )
    await chunk_repository.replace_chunks(
        connection, [office, nonprofits], [unit_vector(0), unit_vector(1)], MODEL
    )

    # "bureau" (singular) matches "Bureaux" through French stemming; short words like "où" are ignored.
    results = await chunk_repository.keyword_search(connection, "Où est le bureau ?", MODEL, top_k=5)
    assert [r["id"] for r in results] == ["MON-01-04"]

    results = await chunk_repository.keyword_search(
        connection, "Travaillez-vous avec des OBNL ?", MODEL, top_k=5
    )
    assert [r["id"] for r in results] == ["MON-05-01"]

    assert await chunk_repository.keyword_search(connection, "?", MODEL, top_k=5) == []


async def test_search_with_embeddings_returns_the_stored_vectors(connection):
    await chunk_repository.replace_chunks(connection, [chunk("MON-01-01")], [unit_vector(3)], MODEL)

    [(found, embedding)] = await chunk_repository.search_with_embeddings(
        connection, unit_vector(3), MODEL, top_k=1
    )

    assert found["id"] == "MON-01-01"
    assert embedding[3] == pytest.approx(1.0)
    assert len(embedding) == DIMENSIONS


async def test_collections_are_isolated(connection):
    await chunk_repository.replace_chunks(
        connection, [chunk("MON-01-01")], [unit_vector(0)], MODEL
    )  # mondria
    upload = "upload-" + "a" * 32
    await chunk_repository.replace_chunks(connection, [chunk("DOC-01-01")], [unit_vector(0)], MODEL, upload)

    assert [r["id"] for r in await chunk_repository.search(connection, unit_vector(0), MODEL, top_k=5)] == [
        "MON-01-01"
    ]
    assert [
        r["id"]
        for r in await chunk_repository.search(connection, unit_vector(0), MODEL, top_k=5, collection=upload)
    ] == ["DOC-01-01"]
    # Re-indexing the Mondria base doesn't delete the upload's rows.
    await chunk_repository.replace_chunks(connection, [chunk("MON-01-01")], [unit_vector(0)], MODEL)
    assert await chunk_repository.count_chunks(connection, MODEL, upload) == 1


async def test_expired_uploads_are_deleted(connection):
    fresh, old = "upload-" + "1" * 32, "upload-" + "2" * 32
    for collection in (fresh, old):
        await chunk_repository.replace_chunks(
            connection, [chunk("DOC-01-01")], [unit_vector(0)], MODEL, collection
        )
        await upload_repository.save_upload(
            connection,
            Upload(
                collection=collection,
                title="T",
                file="t.md",
                kind="markdown",
                size_bytes=10,
                sections=["A"],
                language="fr",
                summary="s",
                suggested_questions=["q"],
                expires_at="",
            ),
        )
    await connection.execute(
        "UPDATE chunks SET created_at = now() - interval '25 hours' WHERE collection = %s", (old,)
    )
    await connection.execute(
        "UPDATE uploads SET created_at = now() - interval '25 hours' WHERE collection = %s", (old,)
    )

    assert await upload_repository.get_upload(connection, old) is None  # expired metadata is already hidden
    await upload_repository.delete_expired_uploads(connection)

    assert await chunk_repository.count_chunks(connection, MODEL, old) == 0
    assert await chunk_repository.count_chunks(connection, MODEL, fresh) == 1
    kept = await upload_repository.get_upload(connection, fresh)
    assert kept is not None and kept["sections"] == ["A"]
