"""Retrieval-Augmented Generation over the Mondria knowledge base."""

import json
import logging
import re
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async

import db
from db import Chunk, Document, ScoredChunk
from openai_client import chat, embed, embedding_model

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent / "mondria-rag-docs"
TOP_K = 4

# Every "## " heading is a boundary; only headings with an ID ("## MON-01-01 — Title") become chunks.
SECTION_PATTERN = re.compile(r"^## (?:(?P<id>[A-Z]+-\d+-\d+) — )?(?P<title>.+)$", re.MULTILINE)

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


_ready_model: str | None = None


async def open_connection() -> psycopg.AsyncConnection:
    """Opens a connection ready for vector queries. The first time per function instance it
    also creates the schema and, if the table has no chunks for the configured embedding
    model, indexes the docs. Changing OPENAI_EMBEDDING_MODEL on Vercel then only needs a
    redeploy: the first request re-embeds the 21 chunks."""
    global _ready_model
    model = embedding_model()
    connection = await db.connect()
    if _ready_model == model:
        await register_vector_async(connection)
        return connection

    await db.ensure_schema(connection)
    if await db.count_chunks(connection, model) == 0:
        logger.warning("No chunks embedded with %s, indexing the docs", model)
        await index_documents(connection, model)
    _ready_model = model
    return connection


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
    raw = await chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": build_context(sources)},
            {"role": "user", "content": question},
        ],
        json_schema=ANSWER_SCHEMA,
    )
    answer = json.loads(raw)["answer"]

    # Sources the answer actually cites, as opposed to the ones that were only retrieved.
    # The model sometimes drops the brackets ("section MON-02-04"), so match the bare ID.
    for source in sources:
        source["cited"] = source["id"] in answer

    return {"answer": answer, "sources": sources}
