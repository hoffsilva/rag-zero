"""The LangChain pipeline with its real components (retrievers, Ensemble, MultiQuery, listwise
rerank, structured output) against a fake OpenAI and fake repository functions."""

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from conftest import openai_transport

from app.domain import Document, ScoredChunk
from app.errors import StrategyNotAvailableError
from app.repositories import chunks as chunk_repository
from app.runtime import open_runtime
from app.services import langchain_rag

DOCUMENT = Document(id="MON-01", title="Entreprise Mondria", file="01.md", source_urls=[])


def chunk(section_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        id=section_id, title="T", text=f"Texte {section_id}", document=DOCUMENT, score=score, cited=False
    )


VECTOR = [chunk("MON-01-01", 0.9), chunk("MON-01-02", 0.8), chunk("MON-01-03", 0.7), chunk("MON-01-04", 0.6)]
KEYWORD = [chunk("MON-01-04", 3.0), chunk("MON-01-03", 2.0)]


class FakeOpenAI:
    """Records requests; answers embeddings, query variants, rankings and structured answers."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.answer = "Deux bureaux [MON-01-04]."
        self.ranking = [3, 0]

    def __call__(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"path": path, **body})
        if path == "/embeddings":
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return {"data": [{"index": i, "embedding": [0.1] * 4} for i in range(len(inputs))]}
        schema = (body.get("response_format") or {}).get("json_schema", {}).get("name")
        if schema == "RankDocuments":
            content = json.dumps({"ranked_document_ids": self.ranking})
        elif schema == "RagAnswer":
            content = json.dumps({"question_language": "French", "answer": self.answer})
        else:  # multi-query variants, one per line
            content = "requête un\nrequête deux"
        return {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-chat",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
            ],
        }


@pytest.fixture
def fake_db(monkeypatch):
    calls: dict[str, list[Any]] = {"search": [], "keyword": []}

    async def search(connection, embedding, model, top_k, collection="mondria"):
        calls["search"].append(top_k)
        return [dict(c) for c in VECTOR[:top_k]]

    async def keyword_search(connection, question, model, top_k, collection="mondria"):
        calls["keyword"].append(question)
        return [dict(c) for c in KEYWORD[:top_k]]

    class FakeConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    async def open_connection(runtime):
        return FakeConnection()

    monkeypatch.setattr(chunk_repository, "search", search)
    monkeypatch.setattr(chunk_repository, "keyword_search", keyword_search)
    monkeypatch.setattr(langchain_rag, "open_connection", open_connection)
    return calls


@asynccontextmanager
async def runtime_with(settings, fake: FakeOpenAI):
    async with open_runtime(settings, openai_transport(fake)) as runtime:
        yield runtime


@pytest.mark.anyio
async def test_vector_keeps_the_cosine_scores_and_marks_citations(settings, fake_db):
    fake = FakeOpenAI()
    async with runtime_with(settings, fake) as runtime:
        result = await langchain_rag.answer_question(runtime, "Où sont les bureaux ?", "vector")

    assert result["pipeline"] == "langchain"
    assert [s["id"] for s in result["sources"]] == ["MON-01-01", "MON-01-02", "MON-01-03", "MON-01-04"]
    assert result["sources"][0]["score"] == 0.9
    assert [s["cited"] for s in result["sources"]] == [False, False, False, True]
    embedding_request = next(r for r in fake.requests if r["path"] == "/embeddings")
    assert embedding_request["dimensions"] == 1536
    chat = next(r for r in fake.requests if r["path"] == "/chat/completions")
    assert chat["max_completion_tokens"] == 500
    assert chat["response_format"]["json_schema"]["strict"] is True
    assert chat["messages"][-1] == {"role": "user", "content": "Où sont les bureaux ?"}
    assert "[MON-01-04] Texte MON-01-04" in chat["messages"][1]["content"]


@pytest.mark.anyio
async def test_hybrid_fuses_vector_and_keyword_with_rrf(settings, fake_db):
    async with runtime_with(settings, FakeOpenAI()) as runtime:
        result = await langchain_rag.answer_question(runtime, "OBNL", "hybrid")

    # Sections found by both lists win: 1/(60+4) + 1/(60+1) for MON-01-04 beats 1/(60+3) + 1/(60+2).
    assert [s["id"] for s in result["sources"]] == ["MON-01-04", "MON-01-03", "MON-01-01", "MON-01-02"]
    assert fake_db["keyword"] == ["OBNL"]


@pytest.mark.anyio
async def test_multi_query_asks_for_variants_and_deduplicates(settings, fake_db):
    fake = FakeOpenAI()
    async with runtime_with(settings, fake) as runtime:
        result = await langchain_rag.answer_question(runtime, "Quanto custa?", "multi_query")

    variants_prompt = fake.requests[0]["messages"][0]["content"]
    assert "language of the documents" in variants_prompt
    assert len(fake_db["search"]) == 3  # two variants + the original question
    ids = [s["id"] for s in result["sources"]]
    assert len(ids) == len(set(ids))


@pytest.mark.anyio
async def test_rerank_orders_the_candidates_with_the_llm(settings, fake_db):
    fake = FakeOpenAI()
    async with runtime_with(settings, fake) as runtime:
        result = await langchain_rag.answer_question(runtime, "q", "rerank")

    assert fake_db["search"] == [10]
    assert [s["id"] for s in result["sources"]] == ["MON-01-04", "MON-01-01"]
    assert [s["score"] for s in result["sources"]] == [1.0, 0.5]


@pytest.mark.anyio
@pytest.mark.parametrize("strategy", ["keyword", "mmr"])
async def test_strategies_without_a_langchain_version_are_refused(settings, fake_db, strategy):
    async with runtime_with(settings, FakeOpenAI()) as runtime:
        with pytest.raises(StrategyNotAvailableError):
            await langchain_rag.answer_question(runtime, "q", strategy)
