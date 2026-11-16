import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import langchain_rag, rag

ANSWER: dict[str, Any] = {
    "answer": "Deux bureaux [MON-01-04].",
    "strategy": "vector",
    "pipeline": "native",
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


@pytest.fixture
def client(settings) -> Iterator[TestClient]:
    """A fresh app per test (its own runtime and MCP server); unexpected errors become 500s."""
    with TestClient(create_app(settings), raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def pipeline(monkeypatch):
    """Replaces the RAG pipeline; records the calls it gets."""
    calls: list[dict] = []

    async def fake_answer(runtime, question, strategy=None, collection="mondria"):
        calls.append({"question": question, "strategy": strategy, "collection": collection})
        return ANSWER

    monkeypatch.setattr(rag, "answer_question", fake_answer)
    return calls


def test_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("question", ["", "   ", "x" * 301])
def test_ask_rejects_empty_or_too_long_questions(client, pipeline, question):
    response = client.post("/api/ask", json={"question": question})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert pipeline == []  # rejected before any embedding or chat call


def test_ask_returns_the_answer_and_sources(client, pipeline):
    response = client.post("/api/ask", json={"question": "  Où sont les bureaux ?  ", "strategy": "hybrid"})

    assert response.status_code == 200
    assert response.json() == ANSWER
    assert pipeline == [{"question": "Où sont les bureaux ?", "strategy": "hybrid", "collection": "mondria"}]


def test_the_langchain_pipeline_is_chosen_per_request(client, pipeline, monkeypatch):
    langchain_calls: list[str | None] = []

    async def fake_langchain(runtime, question, strategy=None, collection="mondria"):
        langchain_calls.append(strategy)
        return {**ANSWER, "pipeline": "langchain"}

    monkeypatch.setattr(langchain_rag, "answer_question", fake_langchain)
    response = client.post(
        "/api/ask", json={"question": "Bonjour", "strategy": "hybrid", "pipeline": "langchain"}
    )

    assert response.status_code == 200
    assert response.json()["pipeline"] == "langchain"
    assert langchain_calls == ["hybrid"]
    assert pipeline == []  # the native pipeline wasn't called


@pytest.mark.parametrize("strategy", ["keyword", "mmr"])
def test_langchain_rejects_the_strategies_it_does_not_offer(client, strategy):
    response = client.post(
        "/api/ask", json={"question": "Bonjour", "strategy": strategy, "pipeline": "langchain"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "strategy_not_available"


def test_ask_rejects_an_unknown_strategy(client, pipeline):
    response = client.post("/api/ask", json={"question": "Bonjour", "strategy": "bm25"})

    assert response.status_code == 422
    fields = response.json()["error"]["fields"]
    assert any(field["location"] == "body.strategy" for field in fields)
    assert pipeline == []


def test_internal_errors_are_generic_and_traceable(client, monkeypatch, caplog):
    async def failing_answer(runtime, question, strategy=None, collection="mondria"):
        raise RuntimeError("OpenAI /embeddings failed (401): secret details")

    monkeypatch.setattr(rag, "answer_question", failing_answer)

    with caplog.at_level(logging.ERROR, logger="app.request"):
        response = client.post("/api/ask", json={"question": "Bonjour"}, headers={"X-Request-ID": "req-42"})

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal server error",
        "error": {"code": "internal_error", "message": "Internal server error", "request_id": "req-42"},
    }
    assert "secret" not in response.text
    assert response.headers["X-Request-ID"] == "req-42"
    assert any(record.request_id == "req-42" and record.exc_info for record in caplog.records)


def test_every_response_has_a_request_id(client):
    generated = client.get("/api/health").headers["X-Request-ID"]
    reused = client.get("/api/health", headers={"X-Request-ID": "abc.123-x"}).headers["X-Request-ID"]
    rejected = client.get("/api/health", headers={"X-Request-ID": "bad id with spaces"}).headers[
        "X-Request-ID"
    ]

    assert len(generated) == 32
    assert reused == "abc.123-x"
    assert rejected != "bad id with spaces" and len(rejected) == 32


def test_access_log_uses_the_route_template_not_the_private_collection(client, caplog):
    collection = "upload-" + "a" * 32
    with caplog.at_level(logging.INFO, logger="app.request"):
        client.get(f"/api/documents/{collection}")

    access_lines = [record.getMessage() for record in caplog.records if record.name == "app.request"]
    assert any('"route": "/api/documents/{collection}"' in line for line in access_lines)
    assert not any(collection in line for line in access_lines)


def test_openapi_documents_the_routes_and_errors(client):
    spec = client.get("/api/openapi.json").json()

    assert set(spec["paths"]) == {"/api/ask", "/api/documents", "/api/documents/{collection}", "/api/health"}
    upload = spec["paths"]["/api/documents"]["post"]
    assert "multipart/form-data" in upload["requestBody"]["content"]
    assert {"413", "415", "422"} <= set(upload["responses"])
    assert spec["components"]["schemas"]["AskRequest"]["examples"][0]["question"]
    assert client.get("/api/docs").status_code == 200


def test_two_apps_are_independent(settings):
    first_app, second_app = create_app(settings), create_app(settings)
    with TestClient(first_app) as first, TestClient(second_app) as second:
        assert first.get("/api/health").status_code == 200
        assert second.get("/api/health").status_code == 200
        assert first_app.state.runtime is not second_app.state.runtime


MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def mcp_request(client: TestClient, method: str, params: dict | None = None) -> dict:
    response = client.post(
        "/api/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
        headers=MCP_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "result" in body, body
    return body["result"]


def test_remote_mcp_endpoint_lists_and_calls_the_tools(client, monkeypatch):
    async def fake_retrieve(runtime, question, top_k=rag.TOP_K, strategy=None, collection="mondria"):
        return [{**ANSWER["sources"][0], "cited": False}]

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    init = mcp_request(
        client,
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
    )
    tools = mcp_request(client, "tools/list")  # stateless: no session ID needed
    result = mcp_request(
        client, "tools/call", {"name": "search_mondria_docs", "arguments": {"query": "bureaux"}}
    )

    assert init["serverInfo"]["name"] == "mondria-rag"
    assert {tool["name"] for tool in tools["tools"]} == {"search_mondria_docs", "ask_mondria"}
    assert result["structuredContent"]["result"][0]["id"] == "MON-01-04"
    assert not result.get("isError")


def test_a_failing_mcp_tool_returns_a_tool_error(client, monkeypatch):
    async def failing_retrieve(runtime, question, top_k=rag.TOP_K, strategy=None, collection="mondria"):
        raise RuntimeError("database down")

    monkeypatch.setattr(rag, "retrieve", failing_retrieve)

    result = mcp_request(client, "tools/call", {"name": "search_mondria_docs", "arguments": {"query": "x"}})

    assert result["isError"] is True
