"""Ingestion of a visitor's uploaded document into its own, temporary collection.

extract text -> chunk -> embed (one batch call) -> store -> describe (summary, language,
suggested questions; one chat call). The Mondria knowledge base is never touched.
"""

import io
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

import db
from db import Chunk, Document, Upload
from openai_client import chat, embed, embedding_model
from rag import open_connection

MAX_BYTES = 1_000_000  # Vercel functions accept up to 4.5 MB per request; 1 MB is plenty for text
MAX_CHUNKS = 100  # bounds the embedding cost and the size of one collection
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 150  # characters repeated between consecutive pieces of a long section
SECTION_LIMIT = 1200  # sections longer than this are cut into overlapping pieces
DOCUMENT_ID = "DOC-01"
KINDS = {".md": "markdown", ".markdown": "markdown", ".txt": "text", ".pdf": "pdf"}
COLLECTION_PATTERN = re.compile(r"^upload-[0-9a-f]{32}$")

HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$", re.MULTILINE)


class DocumentError(Exception):
    """A problem with the uploaded file, reported to the visitor with an HTTP status."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def clean(text: str) -> str:
    """Removes NUL characters, which Postgres text columns reject. Both PDF extraction and the
    model's structured output can produce them (seen in a generated summary)."""
    return text.replace("\x00", "")


def new_collection() -> str:
    # 128 random bits: unguessable, so only the visitor who uploaded the file can query it.
    return f"upload-{uuid.uuid4().hex}"


def is_upload_collection(collection: str) -> bool:
    return bool(COLLECTION_PATTERN.match(collection))


def extract_text(filename: str, data: bytes) -> tuple[str, str]:
    """Returns (text, kind). Raises DocumentError for unsupported, too large or empty files."""
    kind = KINDS.get(PurePath(filename).suffix.lower())
    if kind is None:
        raise DocumentError(415, "Formats acceptés : .md, .txt et .pdf.")
    if len(data) > MAX_BYTES:
        raise DocumentError(413, "Le fichier dépasse 1 Mo.")

    if kind == "pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except (PdfReadError, ValueError, KeyError) as error:
            raise DocumentError(400, "PDF illisible.") from error
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DocumentError(400, "Le fichier texte doit être encodé en UTF-8.") from error

    text = clean(text).replace("\r\n", "\n").strip()
    if not text:
        message = "Aucun texte trouvé (PDF scanné ?)." if kind == "pdf" else "Le fichier est vide."
        raise DocumentError(400, message)
    return text, kind


def split_by_size(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cuts text into pieces of about `size` characters, each starting `overlap` characters before
    the previous one ended, so a sentence cut at a boundary appears whole in one of the pieces.
    Cuts prefer a paragraph break, then a sentence end, then a space."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for separator in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(separator)
                if cut > size // 2:  # don't produce tiny pieces
                    end = start + cut + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def split_sections(text: str) -> list[tuple[str, str]]:
    """Splits on markdown headings (#, ##, ###) into (title, body). Text before the first heading
    becomes an untitled section; a text without headings is a single section."""
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return [("", text)]
    sections = []
    if text[: matches[0].start()].strip():
        sections.append(("", text[: matches[0].start()].strip()))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            sections.append((match.group(2).strip(), body))
    return sections


def chunk_text(text: str, document: Document) -> list[Chunk]:
    """Generic chunking for any document: one chunk per markdown section, long sections (and
    texts without headings) cut by size with overlap. IDs are DOC-01-01, DOC-01-02, ... so
    the answer's citations are detected like the Mondria ones."""
    chunks: list[Chunk] = []
    for title, body in split_sections(text):
        pieces = [body] if len(body) <= SECTION_LIMIT else split_by_size(body)
        for piece in pieces:
            number = len(chunks) + 1
            label = title or first_words(piece)
            chunks.append(
                Chunk(
                    id=f"{DOCUMENT_ID}-{number:02d}",
                    title=label,
                    # Same shape as the Mondria sections: the heading line stays in the embedded text
                    # (it carries meaning), and the UI strips it when showing the section body.
                    text=f"## {title}\n\n{piece}" if title else piece,
                    document=document,
                )
            )
    return chunks


def first_words(text: str, count: int = 8) -> str:
    words = text.split()
    return " ".join(words[:count]) + ("…" if len(words) > count else "")


def document_title(filename: str, text: str, kind: str) -> str:
    if kind == "markdown":
        match = HEADING_PATTERN.search(text)
        if match:
            return match.group(2).strip()
    return PurePath(filename).stem


DESCRIPTION_PROMPT = """Describe the document for someone about to ask questions about it.
Reply with a JSON object with exactly these keys:
- "language": the document's language, written in French (for example "français" or "anglais")
- "summary": two or three sentences in French
- "suggested_questions": a list of three questions the document answers, in the document's language"""


async def describe(text: str) -> dict:
    """Summary, language and suggested questions from the start of the document.

    Uses JSON mode rather than a strict JSON schema: with the strict schema, the model
    intermittently dropped accented letters from long French summaries ("employés" ->
    "employs"), seen on Vercel and locally. The fields are validated here instead."""
    raw = await chat(
        [
            {"role": "system", "content": DESCRIPTION_PROMPT},
            {"role": "user", "content": text[:6000]},
        ],
        json_mode=True,
    )
    description = json.loads(raw)
    questions = description.get("suggested_questions")
    return {
        "language": clean(str(description.get("language", ""))).strip() or "inconnue",
        "summary": clean(str(description.get("summary", ""))).strip(),
        "suggested_questions": [
            clean(q).strip()
            for q in (questions if isinstance(questions, list) else [])
            if isinstance(q, str) and clean(q).strip()
        ][:3],
    }


async def ingest(filename: str, data: bytes) -> Upload:
    text, kind = extract_text(filename, data)
    title = document_title(filename, text, kind)
    document = Document(id=DOCUMENT_ID, title=title, file=PurePath(filename).name, source_urls=[])
    chunks = chunk_text(text, document)
    if len(chunks) > MAX_CHUNKS:
        raise DocumentError(400, f"Document trop long ({len(chunks)} sections, maximum {MAX_CHUNKS}).")

    model = embedding_model()
    embeddings = await embed([chunk["text"] for chunk in chunks], model)
    description = await describe(text)

    collection = new_collection()
    upload = Upload(
        collection=collection,
        title=title,
        file=document["file"],
        kind=kind,
        size_bytes=len(data),
        sections=[chunk["title"] for chunk in chunks],
        language=description["language"],
        summary=description["summary"],
        suggested_questions=description["suggested_questions"],
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=db.UPLOAD_TTL_HOURS)).isoformat(),
    )
    async with await open_connection() as connection:
        await db.delete_expired_uploads(connection)
        async with connection.transaction():
            await db.replace_chunks(connection, chunks, embeddings, model, collection)
            await db.save_upload(connection, upload)
    return upload


async def get_upload(collection: str) -> Upload | None:
    async with await open_connection() as connection:
        return await db.get_upload(connection, collection)
