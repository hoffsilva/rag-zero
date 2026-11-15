import pytest

pytest.importorskip("mcp")

import mcp_server  # noqa: E402
from db import Document, ScoredChunk  # noqa: E402

DOCUMENT = Document(id="MON-01", title="Entreprise Mondria", file="01.md", source_urls=[])


@pytest.mark.anyio
async def test_search_tool_returns_sections_and_clamps_top_k(monkeypatch):
    calls = {}

    async def fake_retrieve(query: str, top_k: int):
        calls["top_k"] = top_k
        return [ScoredChunk(id="MON-01-04", title="Coordonnées", text="...", document=DOCUMENT, score=0.64567, cited=False)]

    monkeypatch.setattr(mcp_server.rag, "retrieve", fake_retrieve)

    results = await mcp_server.search_mondria_docs("offices", top_k=50)

    assert calls["top_k"] == 10
    assert results == [
        {"id": "MON-01-04", "title": "Coordonnées", "document": "Entreprise Mondria", "score": 0.646, "text": "..."}
    ]


@pytest.mark.anyio
async def test_ask_tool_returns_only_cited_sections(monkeypatch):
    async def fake_answer(question: str):
        return {
            "answer": "Deux bureaux [MON-01-04].",
            "sources": [
                ScoredChunk(id="MON-01-04", title="", text="", document=DOCUMENT, score=0.6, cited=True),
                ScoredChunk(id="MON-01-02", title="", text="", document=DOCUMENT, score=0.4, cited=False),
            ],
        }

    monkeypatch.setattr(mcp_server.rag, "answer_question", fake_answer)

    result = await mcp_server.ask_mondria("Où sont les bureaux ?")

    assert result == {"answer": "Deux bureaux [MON-01-04].", "cited_sections": ["MON-01-04"]}
