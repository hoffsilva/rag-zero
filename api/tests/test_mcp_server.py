import pytest

pytest.importorskip("mcp")

import mcp_server  # noqa: E402
from db import Document, ScoredChunk  # noqa: E402

DOCUMENT = Document(id="MON-01", title="Entreprise Mondria", file="01.md", source_urls=[])


@pytest.mark.anyio
async def test_search_tool_returns_sections_and_clamps_top_k(monkeypatch):
    calls = {}

    async def fake_retrieve(query: str, top_k: int, strategy=None):
        calls["top_k"] = top_k
        calls["strategy"] = strategy
        return [
            ScoredChunk(
                id="MON-01-04", title="Coordonnées", text="...", document=DOCUMENT, score=0.64567, cited=False
            )
        ]

    monkeypatch.setattr(mcp_server.rag, "retrieve", fake_retrieve)

    results = await mcp_server.search_mondria_docs("offices", top_k=50, strategy="hybrid")

    assert calls["top_k"] == 10
    assert calls["strategy"] == "hybrid"
    assert results == [
        {
            "id": "MON-01-04",
            "title": "Coordonnées",
            "document": "Entreprise Mondria",
            "score": 0.646,
            "text": "...",
        }
    ]
