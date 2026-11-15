import json

import pytest

import retrieval
from db import Document, ScoredChunk

DOCUMENT = Document(id="MON-01", title="Entreprise Mondria", file="01.md", source_urls=[])


def chunk(section_id: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        id=section_id, title="T", text=f"texte {section_id}", document=DOCUMENT, score=score, cited=False
    )


def ids(chunks: list[ScoredChunk]) -> list[str]:
    return [c["id"] for c in chunks]


def test_rrf_rewards_chunks_ranked_well_in_several_lists():
    vector = [chunk("A"), chunk("B"), chunk("C")]
    keyword = [chunk("C"), chunk("A"), chunk("D")]

    fused = retrieval.reciprocal_rank_fusion([vector, keyword], top_k=4, k=60)

    # A: 1/61 + 1/62, C: 1/63 + 1/61, B: 1/62, D: 1/63
    assert ids(fused) == ["A", "C", "B", "D"]
    assert fused[0]["score"] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_ignores_the_original_score_scales():
    # A huge ts_rank score must not beat a chunk that is first in both lists.
    fused = retrieval.reciprocal_rank_fusion(
        [[chunk("A", 0.9), chunk("B", 0.1)], [chunk("A", 0.01), chunk("B", 99.0)]], top_k=2
    )

    assert ids(fused) == ["A", "B"]


def test_mmr_skips_a_near_duplicate_for_a_new_relevant_chunk():
    query = [1.0, 0.0, 0.0]
    candidates = [
        (chunk("A"), [0.95, 0.312, 0.0]),  # most relevant (cos 0.95)
        (chunk("A-copy"), [0.94, 0.341, 0.0]),  # cos 0.94, but almost identical to A (0.999)
        (chunk("B"), [0.8, -0.2, 0.566]),  # less relevant (0.8), and different from A
    ]

    picked = retrieval.maximal_marginal_relevance(query, candidates, top_k=2, lambda_=0.5)

    assert ids(picked) == ["A", "B"]


def test_mmr_with_lambda_1_is_plain_relevance_order():
    query = [1.0, 0.0]
    candidates = [(chunk("A"), [0.6, 0.8]), (chunk("B"), [1.0, 0.0]), (chunk("C"), [0.8, 0.6])]

    picked = retrieval.maximal_marginal_relevance(query, candidates, top_k=3, lambda_=1.0)

    assert ids(picked) == ["B", "C", "A"]


@pytest.fixture
def chat_returns(monkeypatch):
    sent: list = []

    def set_reply(reply: dict) -> list:
        async def fake_chat(messages, json_schema=None):
            sent.append({"messages": messages, "json_schema": json_schema})
            return json.dumps(reply)

        monkeypatch.setattr(retrieval, "chat", fake_chat)
        return sent

    return set_reply


@pytest.mark.anyio
async def test_rerank_follows_the_llm_order_and_keeps_unlisted_candidates(chat_returns):
    sent = chat_returns({"ranked_ids": ["C", "UNKNOWN", "A", "C"]})  # unknown and repeated IDs

    ranked = await retrieval.rerank_with_llm("q", [chunk("A"), chunk("B"), chunk("C")], top_k=3)

    assert ids(ranked) == ["C", "A", "B"]  # B was left out by the model: kept, at the end
    assert [c["score"] for c in ranked] == pytest.approx([1.0, 2 / 3, 1 / 3])
    assert "[B] texte B" in sent[0]["messages"][-1]["content"]


@pytest.mark.anyio
async def test_multi_query_searches_the_question_and_each_rewrite(chat_returns, monkeypatch):
    chat_returns({"queries": ["bureaux de Mondria", "adresse de Mondria", "  "]})
    embedded: list[list[str]] = []
    searched: list[list[float]] = []

    async def fake_embed(inputs, model):
        embedded.append(inputs)
        return [[float(i)] for i in range(len(inputs))]

    async def fake_search(connection, embedding, model, top_k, collection="mondria"):
        searched.append(embedding)
        return [chunk("MON-01-04")] if embedding == [0.0] else [chunk("MON-01-04"), chunk("MON-01-01")]

    monkeypatch.setattr(retrieval, "embed", fake_embed)
    monkeypatch.setattr(retrieval.db, "search", fake_search)

    results = await retrieval.run("multi_query", None, "Where are the offices?", "m", top_k=2)

    assert embedded == [
        ["Where are the offices?", "bureaux de Mondria", "adresse de Mondria"]
    ]  # blank rewrite dropped
    assert len(searched) == 3
    assert ids(results) == ["MON-01-04", "MON-01-01"]


def test_default_strategy_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "keyword")
    assert retrieval.default_strategy() == "keyword"

    monkeypatch.setenv("RETRIEVAL_STRATEGY", "bm25")
    with pytest.raises(RuntimeError):
        retrieval.default_strategy()
