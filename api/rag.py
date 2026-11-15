"""Retrieval-Augmented Generation over the Mondria knowledge base."""

import json
import logging
import re
from pathlib import Path
from typing import TypedDict

from openai_client import chat, embed, embedding_model

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent / "mondria-rag-docs"
INDEX_FILE = Path(__file__).parent / "data" / "index.json"
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


class IndexedChunk(Chunk):
    embedding: list[float]


class Index(TypedDict):
    model: str
    chunks: list[IndexedChunk]


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


async def build_index(model: str) -> Index:
    """Indexing step: split the docs into sections and embed each one."""
    chunks: list[Chunk] = []
    for file in sorted(DOCS_DIR.glob("*.md")):
        text = file.read_text(encoding="utf-8")
        chunks.extend(split_sections(parse_document(file, text), text))
    embeddings = await embed([chunk["text"] for chunk in chunks], model)
    return Index(
        model=model,
        chunks=[IndexedChunk(**chunk, embedding=embedding) for chunk, embedding in zip(chunks, embeddings)],
    )


_index: Index | None = None


async def get_index() -> Index:
    """Loaded once per function instance. The committed index is reused when it was built with
    the configured embedding model; otherwise the index is rebuilt in memory from the docs, so
    changing OPENAI_EMBEDDING_MODEL on Vercel only needs a redeploy."""
    global _index
    model = embedding_model()
    if _index is None or _index["model"] != model:
        committed: Index | None = json.loads(INDEX_FILE.read_text(encoding="utf-8")) if INDEX_FILE.exists() else None
        # An index built before document metadata existed is treated like a model mismatch.
        if committed and committed["model"] == model and "document" in committed["chunks"][0]:
            _index = committed
        else:
            logger.warning("Committed index does not match %s, rebuilding it in memory", model)
            _index = await build_index(model)
    return _index


def search(index: Index, query_embedding: list[float], top_k: int = TOP_K) -> list[ScoredChunk]:
    # OpenAI embeddings are normalized to length 1, so the dot product is the cosine similarity.
    scored = [
        ScoredChunk(
            id=chunk["id"],
            title=chunk["title"],
            text=chunk["text"],
            document=chunk["document"],
            score=sum(q * c for q, c in zip(query_embedding, chunk["embedding"])),
            cited=False,
        )
        for chunk in index["chunks"]
    ]
    return sorted(scored, key=lambda chunk: chunk["score"], reverse=True)[:top_k]


def build_context(sources: list[ScoredChunk]) -> str:
    """Augmented step: the retrieved chunks become the context of the prompt."""
    return "Context:\n" + "\n\n".join(f"[{chunk['id']}] {chunk['text']}" for chunk in sources)


async def answer_question(question: str) -> dict:
    # Retrieval. The question must be embedded with the same model as the index.
    index = await get_index()
    [query_embedding] = await embed([question], index["model"])
    sources = search(index, query_embedding)

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
    for source in sources:
        source["cited"] = f"[{source['id']}]" in answer

    return {"answer": answer, "sources": sources}
