from fastapi.testclient import TestClient

from app.domain import Document, ScoredChunk
from app.main import create_app
from app.services import rag

DOCUMENT = Document(id="MON-01", title="Entreprise Mondria", file="01.md", source_urls=[])


def test_search_tool_clamps_top_k_and_passes_the_strategy(settings, monkeypatch):
    calls: dict[str, object] = {}

    async def fake_retrieve(runtime, query, top_k=rag.TOP_K, strategy=None, collection="mondria"):
        calls.update(top_k=top_k, strategy=strategy)
        return [
            ScoredChunk(
                id="MON-01-04", title="Coordonnées", text="...", document=DOCUMENT, score=0.64567, cited=False
            )
        ]

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search_mondria_docs",
                    "arguments": {"query": "offices", "top_k": 50, "strategy": "hybrid"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
        )

    assert calls == {"top_k": 10, "strategy": "hybrid"}
    assert response.json()["result"]["structuredContent"]["result"] == [
        {
            "id": "MON-01-04",
            "title": "Coordonnées",
            "document": "Entreprise Mondria",
            "score": 0.646,
            "text": "...",
        }
    ]
