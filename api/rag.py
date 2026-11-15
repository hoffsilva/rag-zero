"""Retrieval-Augmented Generation over the Mondria knowledge base."""

import json
import logging
import re
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async

import db
from db import Chunk, Document, ScoredChunk
from openai_client import ChatRefusal, chat, embed, embedding_model

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent / "mondria-rag-docs"
TOP_K = 4

# Every "## " heading is a boundary; only headings with an ID ("## MON-01-01 — Title") become chunks.
SECTION_PATTERN = re.compile(r"^## (?:(?P<id>[A-Z]+-\d+-\d+) — )?(?P<title>.+)$", re.MULTILINE)

# A section ID as a whole token: "[MON-01-04]" or "MON-01-04", but not inside "MON-01-040".
CITED_ID_PATTERN = re.compile(r"(?<![\w-])[A-Z]+-\d+-\d+(?![\w-])")

SYSTEM_PROMPT = """You answer questions about the company Mondria using a knowledge base written in French.
Answer only from the provided context.
Cite the IDs of the sections you used in square brackets, for example [MON-01-04].
If the information is not in the context, say clearly that it is not available. Never make anything up.
Always answer in the language of the user's question (for example Portuguese, English or French), \
even though the context is in French. This also applies when the information is not available. Be concise."""

ANSWER_SCHEMA = {
    "name": "rag_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "question_language": {"type": "string", "description": "Language the user's question is written in."},
            "answer": {"type": "string", "description": "The answer, written in question_language, citing section IDs like [MON-01-04]."},
        },
        "required": ["question_language", "answer"],
        "additionalProperties": False,
    },
}


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
        sections.append(Chunk(
            id=match["id"],
            title=match["title"].strip(),
            text=text[match.start():end].strip(),
            document=document,
        ))
    return sections


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for file in sorted(DOCS_DIR.glob("*.md")):
        text = file.read_text(encoding="utf-8")
        chunks.extend(split_sections(parse_document(file, text), text))
    return chunks


async def index_documents(connection: psycopg.AsyncConnection, model: str) -> int:
    """Indexing step: split the docs into sections, embed them in one batch and store them."""
    chunks = load_chunks()
    embeddings = await embed([chunk["text"] for chunk in chunks], model)
    await db.replace_chunks(connection, chunks, embeddings, model)
    return len(chunks)


# Arbitrary constant identifying this app's indexing lock in pg_advisory_xact_lock.
INDEX_LOCK_KEY = 7_201_204

_ready_model: str | None = None


async def prepare(connection: psycopg.AsyncConnection, model: str) -> None:
    """Creates the schema and indexes the docs if the table has no chunks for the model.

    Runs in one transaction under a Postgres advisory lock, so when several cold instances
    start together only the first one indexes; the others wait, then see the chunks and skip.
    """
    async with connection.transaction():
        await connection.execute("SELECT pg_advisory_xact_lock(%s)", (INDEX_LOCK_KEY,))
        await db.ensure_schema(connection)
        if await db.count_chunks(connection, model) == 0:
            logger.warning("No chunks embedded with %s, indexing the docs", model)
            await index_documents(connection, model)
    # Leaving the block committed the transaction (and released the lock).


async def open_connection() -> psycopg.AsyncConnection:
    """Opens a connection ready for vector queries. The first time per function instance it
    also prepares the database (see prepare). Changing OPENAI_EMBEDDING_MODEL on Vercel then
    only needs a redeploy: the first request embeds the 21 chunks with the new model."""
    global _ready_model
    model = embedding_model()
    connection = await db.connect()
    try:
        if _ready_model == model:
            await register_vector_async(connection)
        else:
            await prepare(connection, model)
            _ready_model = model  # only after the indexing is committed
        return connection
    except BaseException:
        await connection.close()  # also on cancellation, so no connection or lock is left behind
        raise


def build_context(sources: list[ScoredChunk]) -> str:
    """Augmented step: the retrieved chunks become the context of the prompt."""
    return "Context:\n" + "\n\n".join(f"[{chunk['id']}] {chunk['text']}" for chunk in sources)


async def retrieve(question: str, top_k: int = TOP_K) -> list[ScoredChunk]:
    """Retrieval step. The question must be embedded with the same model as the chunks."""
    model = embedding_model()
    [query_embedding] = await embed([question], model)
    async with await open_connection() as connection:
        return await db.search(connection, query_embedding, model, top_k)


async def answer_question(question: str) -> dict:
    sources = await retrieve(question)

    # Generation. With a French context, the model tends to answer in French whatever the
    # question's language. Making it name the question's language first, in a structured
    # response, keeps the answer in that language.
    try:
        raw = await chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": build_context(sources)},
                {"role": "user", "content": question},
            ],
            json_schema=ANSWER_SCHEMA,
        )
    except ChatRefusal as refusal:
        # A refusal is an answer, not a server error: show it, with nothing cited.
        return {"answer": str(refusal), "sources": sources}
    answer = json.loads(raw)["answer"]

    # Sources the answer actually cites, as opposed to the ones that were only retrieved.
    # IDs are compared as whole tokens; the model sometimes drops the brackets.
    cited_ids = set(CITED_ID_PATTERN.findall(answer))
    for source in sources:
        source["cited"] = source["id"] in cited_ids

    return {"answer": answer, "sources": sources}
