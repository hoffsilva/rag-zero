"""The Mondria knowledge base: parsing, indexing, and database preparation."""

import logging
import re
from pathlib import Path

from app.clients.openai import OpenAIClient
from app.domain import Chunk, Document
from app.paths import DOCS_DIR
from app.repositories import chunks as chunk_repository
from app.repositories.database import Connection, ensure_schema, register_vector
from app.runtime import Runtime

logger = logging.getLogger(__name__)

# Every "## " heading is a boundary; only headings with an ID ("## MON-01-01 — Title") become chunks.
SECTION_PATTERN = re.compile(r"^## (?:(?P<id>[A-Z]+-\d+-\d+) — )?(?P<title>.+)$", re.MULTILINE)
# Arbitrary constant identifying this app's indexing lock in pg_advisory_xact_lock.
INDEX_LOCK_KEY = 7_201_204


class EmptyCorpusError(RuntimeError):
    """No document or section found: indexing would wipe the knowledge base."""


def parse_document(file: Path, text: str) -> Document:
    """Reads id, title and source_urls from the doc's YAML frontmatter (flat keys and one list)."""
    frontmatter = text.split("---", 2)[1] if text.startswith("---") else ""
    fields = dict(re.findall(r"^(\w+):\s*\"?(.*?)\"?\s*$", frontmatter, re.MULTILINE))
    return Document(
        id=fields.get("id", file.stem),
        title=fields.get("title", file.stem),
        file=file.name,
        source_urls=re.findall(r"^\s+-\s+(https?://\S+)$", frontmatter, re.MULTILINE),
    )


def split_sections(document: Document, text: str) -> list[Chunk]:
    matches = list(SECTION_PATTERN.finditer(text))
    sections: list[Chunk] = []
    for position, match in enumerate(matches):
        if not match["id"]:
            continue  # "Documents associés", "Sources": boundary only, not a chunk
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        sections.append(
            Chunk(
                id=match["id"],
                title=match["title"].strip(),
                text=text[match.start() : end].strip(),
                document=document,
            )
        )
    return sections


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """All sections of the knowledge base. Raises instead of returning nothing: replacing the
    stored chunks with an empty list would delete the whole base."""
    files = sorted(docs_dir.glob("*.md"))
    chunks: list[Chunk] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        chunks.extend(split_sections(parse_document(file, text), text))
    if not chunks:
        raise EmptyCorpusError(f"No sections found in {len(files)} Markdown files under {docs_dir}")
    return chunks


async def index_documents(openai: OpenAIClient, connection: Connection, model: str) -> int:
    """Indexing step: split the docs into sections, embed them in one batch and store them."""
    chunks = load_chunks()
    embeddings = await openai.embed([chunk["text"] for chunk in chunks], model)
    await chunk_repository.replace_chunks(connection, chunks, embeddings, model)
    return len(chunks)


async def prepare(openai: OpenAIClient, connection: Connection, model: str) -> None:
    """Creates the schema and indexes the docs if the table has no chunks for the model.

    Runs in one transaction under a Postgres advisory lock, so when several cold instances
    start together only the first one indexes; the others wait, then see the chunks and skip.
    """
    async with connection.transaction():
        await connection.execute("SELECT pg_advisory_xact_lock(%s)", (INDEX_LOCK_KEY,))
        await ensure_schema(connection)
        if await chunk_repository.count_chunks(connection, model) == 0:
            logger.warning("No chunks embedded with %s, indexing the docs", model)
            await index_documents(openai, connection, model)
    # Leaving the block committed the transaction (and released the lock).


async def open_connection(runtime: Runtime) -> Connection:
    """Opens a connection ready for vector queries. The first time per runtime it also prepares
    the database (see prepare). Changing OPENAI_EMBEDDING_MODEL on Vercel then only needs a
    redeploy: the first request embeds the 21 chunks with the new model."""
    model = runtime.openai.embedding_model
    connection = await runtime.database.connect()
    try:
        if runtime.database.ready_model == model:
            await register_vector(connection)
        else:
            await prepare(runtime.openai, connection, model)
            runtime.database.ready_model = model  # only after the indexing is committed
        return connection
    except BaseException:
        await connection.close()  # also on cancellation, so no connection or lock is left behind
        raise
