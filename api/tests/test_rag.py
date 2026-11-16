import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.clients.openai import ChatRefusal
from app.domain import Document, ScoredChunk
from app.repositories import chunks as chunk_repository
from app.services import indexing, rag

DOCUMENT = Document(
    id="MON-01", title="Entreprise Mondria", file="01.md", source_urls=["https://mondria.com/"]
)


def source(section_id: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        id=section_id,
        title="T",
        text=f"## {section_id} — T\n\nTexte.",
        document=DOCUMENT,
        score=score,
        cited=False,
    )


class FakeOpenAI:
    """Records chat calls and answers with a fixed reply (or a refusal when reply is None)."""

    embedding_model = "text-embedding-3-small"

    def __init__(self) -> None:
        self.reply: str | None = "ok"
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[dict[str, str]], json_schema: Any = None, json_mode: bool = False
    ) -> str:
        self.calls.append({"messages": messages, "json_schema": json_schema})
        if self.reply is None:
            raise ChatRefusal("I can't help with that.")
        return json.dumps({"question_language": "French", "answer": self.reply})


@pytest.fixture
def runtime(settings, monkeypatch):
    """A runtime with a fake model, and retrieval replaced by three fixed sections."""

    async def fake_retrieve(runtime, question, top_k=rag.TOP_K, strategy=None, collection="mondria"):
        return [source("MON-01-04", 0.6), source("MON-01-02", 0.4), source("MON-02-04", 0.3)]

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    return SimpleNamespace(settings=settings, openai=FakeOpenAI(), database=None)


@pytest.mark.anyio
async def test_generation_receives_the_question_and_the_retrieved_sections(runtime):
    await rag.answer_question(runtime, "Quanto custa o suporte?")

    [call] = runtime.openai.calls
    prompt = "\n".join(message["content"] for message in call["messages"])
    assert call["messages"][-1] == {"role": "user", "content": "Quanto custa o suporte?"}
    for section_id in ["MON-01-04", "MON-01-02", "MON-02-04"]:
        assert f"[{section_id}]" in prompt
    assert call["json_schema"] is not None  # structured answer that names the language first


@pytest.mark.anyio
async def test_answer_marks_cited_sources_with_or_without_brackets(runtime):
    runtime.openai.reply = "Deux bureaux [MON-01-04]. Aucun tarif fixe (section MON-02-04)."

    result = await rag.answer_question(runtime, "Où sont les bureaux ?")

    cited = {s["id"]: s["cited"] for s in result["sources"]}
    assert cited == {"MON-01-04": True, "MON-01-02": False, "MON-02-04": True}


@pytest.mark.anyio
async def test_citations_are_matched_as_whole_ids(runtime):
    runtime.openai.reply = "Voir [MON-01-040] et MON-02-04x."  # neither is a retrieved ID

    result = await rag.answer_question(runtime, "q")

    assert not any(s["cited"] for s in result["sources"])


@pytest.mark.anyio
async def test_answer_without_citation_marks_nothing(runtime):
    runtime.openai.reply = "Cette information n'est pas disponible."

    result = await rag.answer_question(runtime, "Qui est le PDG ?")

    assert result["answer"] == "Cette information n'est pas disponible."
    assert not any(s["cited"] for s in result["sources"])


@pytest.mark.anyio
async def test_a_model_refusal_is_returned_as_the_answer(runtime):
    runtime.openai.reply = None

    result = await rag.answer_question(runtime, "q")

    assert result["answer"] == "I can't help with that."
    assert not any(s["cited"] for s in result["sources"])


@pytest.mark.anyio
async def test_default_strategy_comes_from_the_settings(runtime):
    result = await rag.answer_question(runtime, "q")

    assert result["strategy"] == "rerank"


# --- database preparation (open_connection) ------------------------------------------------


class FakeTransaction:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    async def __aenter__(self) -> "FakeTransaction":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.state["committed" if exc_type is None else "rolled_back"] += 1


class FakeConnection:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self.state)

    async def execute(self, query: str, params: Any = None) -> None:
        if "pg_advisory_xact_lock" in query:
            self.state["locked"] += 1

    async def close(self) -> None:
        self.state["closed"] += 1


class FakeDatabase:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.ready_model: str | None = None

    async def connect(self) -> FakeConnection:
        return FakeConnection(self.state)


@pytest.fixture
def database(settings, monkeypatch):
    """Fakes the database layer and records what open_connection does."""
    state: dict[str, Any] = {
        "rows": {},
        "indexed": [],
        "schema_created": 0,
        "registered": 0,
        "locked": 0,
        "committed": 0,
        "rolled_back": 0,
        "closed": 0,
        "fail_indexing": False,
    }

    async def ensure_schema(connection):
        state["schema_created"] += 1

    async def count_chunks(connection, model, collection="mondria"):
        return state["rows"].get(model, 0)

    async def index_documents(openai, connection, model):
        if state["fail_indexing"]:
            raise RuntimeError("embeddings failed")
        state["indexed"].append(model)
        state["rows"][model] = 21  # models coexist
        return 21

    async def register_vector(connection):
        state["registered"] += 1

    monkeypatch.setattr(indexing, "ensure_schema", ensure_schema)
    monkeypatch.setattr(chunk_repository, "count_chunks", count_chunks)
    monkeypatch.setattr(indexing, "index_documents", index_documents)
    monkeypatch.setattr(indexing, "register_vector", register_vector)
    openai = SimpleNamespace(embedding_model="text-embedding-3-small")
    state["runtime"] = SimpleNamespace(settings=settings, openai=openai, database=FakeDatabase(state))
    return state


@pytest.mark.anyio
async def test_first_request_indexes_when_the_table_is_empty(database):
    runtime = database["runtime"]

    await indexing.open_connection(runtime)
    await indexing.open_connection(runtime)  # same runtime, same model: no schema check, no indexing

    assert database["indexed"] == ["text-embedding-3-small"]
    assert database["schema_created"] == 1
    assert database["registered"] == 1
    assert database["locked"] == 1  # indexing ran under the advisory lock
    assert database["committed"] == 1


@pytest.mark.anyio
async def test_existing_chunks_for_the_model_are_reused(database):
    database["rows"] = {"text-embedding-3-small": 21}

    await indexing.open_connection(database["runtime"])

    assert database["indexed"] == []


@pytest.mark.anyio
async def test_changing_the_embedding_model_reindexes(database):
    runtime = database["runtime"]
    database["rows"] = {"text-embedding-3-small": 21}
    await indexing.open_connection(runtime)

    runtime.openai.embedding_model = "text-embedding-3-large"  # the Vercel variable changed
    await indexing.open_connection(runtime)

    assert database["indexed"] == ["text-embedding-3-large"]


@pytest.mark.anyio
async def test_failed_indexing_closes_the_connection_and_stays_not_ready(database):
    database["fail_indexing"] = True

    with pytest.raises(RuntimeError):
        await indexing.open_connection(database["runtime"])

    assert database["rolled_back"] == 1
    assert database["closed"] == 1
    assert database["runtime"].database.ready_model is None  # the next request retries


def test_indexing_refuses_an_empty_corpus(tmp_path):
    (tmp_path / "empty.md").write_text("# Pas de section numérotée\n")

    with pytest.raises(indexing.EmptyCorpusError):
        indexing.load_chunks(tmp_path)
