import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

ANSWER = {
    "answer": "Deux bureaux [MON-01-04].",
    "strategy": "vector",
    "sources": [
        {
            "id": "MON-01-04",
            "title": "Coordonnées",
            "text": "## MON-01-04 — Coordonnées\n\nBureaux publiés : ...",
            "document": {"id": "MON-01", "title": "Entreprise Mondria", "file": "01.md", "source_urls": []},
            "score": 0.645,
            "cited": True,
        }
    ],
}


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("question", ["", "   ", "x" * 301])
def test_ask_rejects_empty_or_too_long_questions(question, monkeypatch):
    calls = []

    async def pipeline_must_not_run(question: str, strategy=None):
        calls.append(question)
        return ANSWER

    monkeypatch.setattr(main, "answer_question", pipeline_must_not_run)

    response = client.post("/api/ask", json={"question": question})

    assert response.status_code == 422
    assert calls == []  # rejected before any embedding or chat call


def test_ask_returns_the_answer_and_sources(monkeypatch):
    received = {}

    async def fake_answer(question: str, strategy=None):
        received["question"] = question
        received["strategy"] = strategy
        return ANSWER

    monkeypatch.setattr(main, "answer_question", fake_answer)

    response = client.post("/api/ask", json={"question": "  Où sont les bureaux ?  ", "strategy": "hybrid"})

    assert response.status_code == 200
    assert response.json() == ANSWER
    assert received["question"] == "Où sont les bureaux ?"  # stripped before the pipeline
    assert received["strategy"] == "hybrid"


def test_ask_rejects_an_unknown_strategy(monkeypatch):
    async def pipeline_must_not_run(question: str, strategy=None):
        raise AssertionError("pipeline ran")

    monkeypatch.setattr(main, "answer_question", pipeline_must_not_run)

    response = client.post("/api/ask", json={"question": "Bonjour", "strategy": "bm25"})

    assert response.status_code == 422


def test_ask_hides_internal_errors(monkeypatch):
    async def failing_answer(question: str, strategy=None):
        raise RuntimeError("OpenAI /embeddings failed (401): secret details")

    monkeypatch.setattr(main, "answer_question", failing_answer)

    response = client.post("/api/ask", json={"question": "Bonjour"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not answer the question"}
    assert "secret" not in response.text


MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def mcp_request(client: TestClient, method: str, params: dict | None = None) -> dict:
    response = client.post(
        "/api/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
        headers=MCP_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def test_remote_mcp_endpoint_lists_the_tools():
    # Entering the client runs the lifespan that starts the MCP session manager, as Vercel does.
    with TestClient(main.app) as mcp_client:
        init = mcp_request(
            mcp_client,
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
        )
        tools = mcp_request(mcp_client, "tools/list")  # stateless: no session ID needed

    assert init["serverInfo"]["name"] == "mondria-rag"
    assert {tool["name"] for tool in tools["tools"]} == {"search_mondria_docs", "ask_mondria"}
