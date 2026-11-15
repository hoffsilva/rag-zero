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
    """Replaces retrieval and the chat call. The returned function sets the model's answer
    and returns a dict that records what the chat call received."""
    received: dict = {}

    async def fake_retrieve(question: str, top_k: int = rag.TOP_K, strategy=None):
        return [source("MON-01-04", 0.6), source("MON-01-02", 0.4), source("MON-02-04", 0.3)]

    def set_answer(answer: str) -> dict:
        async def fake_chat(messages, json_schema=None):
            received["messages"] = messages
            received["json_schema"] = json_schema
            return json.dumps({"question_language": "French", "answer": answer})

        monkeypatch.setattr(rag, "chat", fake_chat)
        return received

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    return set_answer


@pytest.mark.anyio
async def test_generation_receives_the_question_and_the_retrieved_sections(model_answers):
    received = model_answers("ok")

    await rag.answer_question("Quanto custa o suporte?")

    prompt = "\n".join(message["content"] for message in received["messages"])
    assert received["messages"][-1] == {"role": "user", "content": "Quanto custa o suporte?"}
    for section_id in ["MON-01-04", "MON-01-02", "MON-02-04"]:
        assert f"[{section_id}]" in prompt
    assert received["json_schema"] is not None  # structured answer that names the language first


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


class FakeTransaction:
    def __init__(self, state: dict):
        self.state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.state["committed" if exc_type is None else "rolled_back"] += 1


class FakeConnection:
    def __init__(self, state: dict):
        self.state = state

    def transaction(self):
        return FakeTransaction(self.state)

    async def execute(self, query: str, params=None):
        if "pg_advisory_xact_lock" in query:
            self.state["locked"] += 1

    async def close(self) -> None:
        self.state["closed"] += 1


@pytest.fixture
def database(monkeypatch):
    """Fakes the database layer and records what open_connection does."""
    state = {
        "rows": {}, "indexed": [], "schema_created": 0, "registered": 0,
        "locked": 0, "committed": 0, "rolled_back": 0, "closed": 0, "fail_indexing": False,
    }

    async def connect():
        return FakeConnection(state)

    async def ensure_schema(connection):
        state["schema_created"] += 1

    async def count_chunks(connection, model):
        return state["rows"].get(model, 0)

    async def index_documents(connection, model):
        if state["fail_indexing"]:
            raise RuntimeError("embeddings failed")
        state["indexed"].append(model)
        state["rows"][model] = 21  # models coexist
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
    assert database["locked"] == 1  # indexing ran under the advisory lock
    assert database["committed"] == 1


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


@pytest.mark.anyio
async def test_failed_indexing_closes_the_connection_and_stays_not_ready(database, monkeypatch):
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    database["fail_indexing"] = True

    with pytest.raises(RuntimeError):
        await rag.open_connection()

    assert database["rolled_back"] == 1
    assert database["closed"] == 1
    assert rag._ready_model is None  # the next request retries instead of trusting a rolled-back index


@pytest.mark.anyio
async def test_citations_are_matched_as_whole_ids(model_answers):
    model_answers("Voir [MON-01-040] et MON-02-04x.")  # neither is a retrieved ID

    result = await rag.answer_question("q")

    assert not any(s["cited"] for s in result["sources"])


@pytest.mark.anyio
async def test_a_model_refusal_is_returned_as_the_answer(monkeypatch):
    async def fake_retrieve(question: str, top_k: int = rag.TOP_K, strategy=None):
        return [source("MON-01-04")]

    async def refusing_chat(messages, json_schema=None):
        raise rag.ChatRefusal("I can't help with that.")

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "chat", refusing_chat)

    result = await rag.answer_question("q")

    assert result["answer"] == "I can't help with that."
    assert not result["sources"][0]["cited"]
