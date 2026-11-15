import json

import pytest

import rag
from db import Document, ScoredChunk

DOCUMENT = Document(id="MON-01", title="Entreprise Mondria", file="01.md", source_urls=["https://mondria.com/"])


def source(section_id: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        id=section_id, title="T", text=f"## {section_id} — T\n\nTexte.", document=DOCUMENT, score=score, cited=False
    )


@pytest.fixture
def model_answers(monkeypatch):
    """Replaces retrieval and the chat call; the returned function sets the model's answer."""

    async def fake_retrieve(question: str, top_k: int = rag.TOP_K):
        return [source("MON-01-04", 0.6), source("MON-01-02", 0.4), source("MON-02-04", 0.3)]

    def set_answer(answer: str) -> None:
        async def fake_chat(messages, json_schema=None):
            return json.dumps({"question_language": "French", "answer": answer})

        monkeypatch.setattr(rag, "chat", fake_chat)

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    return set_answer


@pytest.mark.anyio
async def test_answer_marks_cited_sources_with_or_without_brackets(model_answers):
    model_answers("Deux bureaux [MON-01-04]. Aucun tarif fixe (section MON-02-04).")

    result = await rag.answer_question("Où sont les bureaux ?")

    cited = {s["id"]: s["cited"] for s in result["sources"]}
    assert cited == {"MON-01-04": True, "MON-01-02": False, "MON-02-04": True}


@pytest.mark.anyio
async def test_answer_without_citation_marks_nothing(model_answers):
    model_answers("Cette information n'est pas disponible.")

    result = await rag.answer_question("Qui est le PDG ?")

    assert result["answer"] == "Cette information n'est pas disponible."
    assert not any(s["cited"] for s in result["sources"])


class FakeConnection:
    async def close(self) -> None:
        pass


@pytest.fixture
def database(monkeypatch):
    """Fakes the database layer and records what open_connection does."""
    state = {"rows": {}, "indexed": [], "schema_created": 0, "registered": 0}

    async def connect():
        return FakeConnection()

    async def ensure_schema(connection):
        state["schema_created"] += 1

    async def count_chunks(connection, model):
        return state["rows"].get(model, 0)

    async def index_documents(connection, model):
        state["indexed"].append(model)
        state["rows"] = {model: 21}  # indexing replaces rows of any other model
        return 21

    async def register_vector(connection):
        state["registered"] += 1

    monkeypatch.setattr(rag.db, "connect", connect)
    monkeypatch.setattr(rag.db, "ensure_schema", ensure_schema)
    monkeypatch.setattr(rag.db, "count_chunks", count_chunks)
    monkeypatch.setattr(rag, "index_documents", index_documents)
    monkeypatch.setattr(rag, "register_vector_async", register_vector)
    monkeypatch.setattr(rag, "_ready_model", None)
    return state


@pytest.mark.anyio
async def test_first_request_indexes_when_the_table_is_empty(database, monkeypatch):
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    await rag.open_connection()
    await rag.open_connection()  # same instance, same model: no schema check, no indexing

    assert database["indexed"] == ["text-embedding-3-small"]
    assert database["schema_created"] == 1
    assert database["registered"] == 1


@pytest.mark.anyio
async def test_existing_chunks_for_the_model_are_reused(database, monkeypatch):
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    database["rows"] = {"text-embedding-3-small": 21}

    await rag.open_connection()

    assert database["indexed"] == []


@pytest.mark.anyio
async def test_changing_the_embedding_model_reindexes(database, monkeypatch):
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    database["rows"] = {"text-embedding-3-small": 21}
    await rag.open_connection()

    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")  # the Vercel variable changed
    await rag.open_connection()

    assert database["indexed"] == ["text-embedding-3-large"]
