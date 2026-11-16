import json

import pytest
from conftest import openai_transport
from fastapi.testclient import TestClient

from app.domain import Document
from app.errors import AppError
from app.main import create_app
from app.repositories import chunks as chunk_repository
from app.repositories import uploads as upload_repository
from app.services import ingestion as documents

DOCUMENT = Document(id="DOC-01", title="Guide", file="guide.md", source_urls=[])


def make_pdf(text: str | None) -> bytes:
    """A minimal one-page PDF; text=None gives a page with no text (like a scanned image)."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return pdf


# --- extraction -----------------------------------------------------------------------------


def test_extracts_markdown_and_text_as_utf8():
    assert documents.extract_text("notes.md", "# Titre\r\n\r\nCafé".encode()) == (
        "# Titre\n\nCafé",
        "markdown",
    )
    assert documents.extract_text("notes.TXT", b"  hello  ") == ("hello", "text")


def test_extracts_text_from_a_pdf():
    text, kind = documents.extract_text("rapport.pdf", make_pdf("Bonjour depuis un PDF"))

    assert kind == "pdf"
    assert "Bonjour depuis un PDF" in text


@pytest.mark.parametrize(
    ("filename", "data", "status"),
    [
        ("virus.exe", b"MZ", 415),
        ("big.txt", b"a" * (documents.MAX_BYTES + 1), 413),
        ("empty.txt", b"   ", 400),
        ("latin1.txt", "café".encode("latin-1"), 400),
        ("scan.pdf", make_pdf(None), 400),
        ("broken.pdf", b"%PDF-1.4 not really", 400),
    ],
)
def test_rejects_unsupported_too_large_or_empty_files(filename, data, status):
    with pytest.raises(AppError) as error:
        documents.extract_text(filename, data)

    assert error.value.status_code == status


# --- chunking -------------------------------------------------------------------------------


def test_one_chunk_per_markdown_section_with_citable_ids():
    text = "# Guide\n\nIntro.\n\n## Horaires\n\nTrois jours.\n\n### Exceptions\n\nAucune.\n\n## Vide\n"

    chunks = documents.chunk_text(text, DOCUMENT)

    assert [c["id"] for c in chunks] == ["DOC-01-01", "DOC-01-02", "DOC-01-03"]  # empty section dropped
    assert [c["title"] for c in chunks] == ["Guide", "Horaires", "Exceptions"]
    assert chunks[1]["text"] == "## Horaires\n\nTrois jours."


def test_long_sections_are_cut_with_overlap():
    sentences = " ".join(f"Phrase numéro {i}." for i in range(200))

    chunks = documents.chunk_text(f"## Long\n\n{sentences}", DOCUMENT)

    assert len(chunks) > 1
    assert all(c["title"] == "Long" for c in chunks)
    bodies = [c["text"].removeprefix("## Long\n\n") for c in chunks]
    assert all(len(body) <= documents.CHUNK_SIZE for body in bodies)
    # Consecutive pieces share text, so a sentence cut at a boundary is whole in one of them.
    assert bodies[0][-60:].split()[-1] in bodies[1]


def test_text_without_headings_is_cut_by_size_and_titled_by_its_first_words():
    chunks = documents.chunk_text("Un paragraphe sans titre. " * 100, DOCUMENT)

    assert len(chunks) > 1
    assert chunks[0]["title"].startswith("Un paragraphe sans titre")


def test_split_by_size_prefers_sentence_boundaries():
    text = "A" * 500 + ". " + "B" * 500

    first, *_ = documents.split_by_size(text, size=800, overlap=150)

    assert first.endswith(".")


# --- API ------------------------------------------------------------------------------------


def fake_openai(path: str, body: dict) -> dict:
    if path == "/embeddings":
        return {"data": [{"index": i, "embedding": [0.0] * 1536} for i, _ in enumerate(body["input"])]}
    # The description call. A NUL character, as the model once produced in a summary:
    # Postgres would reject it.
    content = json.dumps(
        {"language": "français", "summary": "Un\u0000 guide.", "suggested_questions": ["Q1 ?", "", "Q2 ?"]}
    )
    return {"choices": [{"finish_reason": "stop", "message": {"content": content, "refusal": None}}]}


@pytest.fixture
def saved(monkeypatch):
    """Fakes the database writes of the upload and records them."""
    records: dict = {}

    class FakeConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def transaction(self):
            return self

    async def fake_open_connection(runtime):
        return FakeConnection()

    async def fake_replace(connection, chunks, embeddings, model, collection):
        records["chunks"] = chunks
        records["collection"] = collection

    async def fake_save(connection, upload):
        records["upload"] = upload

    async def noop(connection):
        pass

    monkeypatch.setattr(documents, "open_connection", fake_open_connection)
    monkeypatch.setattr(chunk_repository, "replace_chunks", fake_replace)
    monkeypatch.setattr(upload_repository, "save_upload", fake_save)
    monkeypatch.setattr(upload_repository, "delete_expired_uploads", noop)
    return records


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, openai_transport(fake_openai))) as test_client:
        yield test_client


def test_upload_creates_a_private_collection(client, saved):
    markdown = "# Guide du télétravail\n\n## Horaires\n\nTrois jours.\n\n## Sécurité\n\nVPN obligatoire."

    response = client.post("/api/documents", files={"file": ("guide.md", markdown.encode(), "text/markdown")})

    assert response.status_code == 200, response.text
    body = response.json()
    assert documents.is_upload_collection(body["collection"])
    assert body["title"] == "Guide du télétravail"
    assert body["sections"] == ["Horaires", "Sécurité"]
    assert body["suggested_questions"] == ["Q1 ?", "Q2 ?"]  # blank question dropped
    assert body["summary"] == "Un guide."  # NUL removed
    assert saved["collection"] == body["collection"]
    assert [c["id"] for c in saved["chunks"]] == ["DOC-01-01", "DOC-01-02"]


def test_upload_reports_file_problems_with_their_status(client, saved):
    response = client.post("/api/documents", files={"file": ("photo.png", b"\x89PNG", "image/png")})

    assert response.status_code == 415
    body = response.json()
    assert body["error"]["code"] == "unsupported_file_type"
    assert "Formats acceptés" in body["detail"]  # kept for compatibility
    assert "upload" not in saved


def test_unknown_or_malformed_collections_are_not_found(client, monkeypatch):
    async def no_upload(runtime, collection):
        return None

    monkeypatch.setattr(documents, "get_upload", no_upload)

    assert client.get("/api/documents/upload-" + "0" * 32).status_code == 404
    assert client.get("/api/documents/mondria").json()["error"]["code"] == "document_not_found"
    response = client.post("/api/ask", json={"question": "q", "collection": "upload-" + "0" * 32})
    assert response.status_code == 404
    assert client.post("/api/ask", json={"question": "q", "collection": "someone-else"}).status_code == 422
