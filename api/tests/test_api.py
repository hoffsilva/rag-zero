import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

ANSWER = {
    "answer": "Deux bureaux [MON-01-04].",
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

    async def pipeline_must_not_run(question: str):
        calls.append(question)
        return ANSWER

    monkeypatch.setattr(main, "answer_question", pipeline_must_not_run)

    response = client.post("/api/ask", json={"question": question})

    assert response.status_code == 422
    assert calls == []  # rejected before any embedding or chat call


def test_ask_returns_the_answer_and_sources(monkeypatch):
    received = {}

    async def fake_answer(question: str):
        received["question"] = question
        return ANSWER

    monkeypatch.setattr(main, "answer_question", fake_answer)

    response = client.post("/api/ask", json={"question": "  Où sont les bureaux ?  "})

    assert response.status_code == 200
    assert response.json() == ANSWER
    assert received["question"] == "Où sont les bureaux ?"  # stripped before the pipeline


def test_ask_hides_internal_errors(monkeypatch):
    async def failing_answer(question: str):
        raise RuntimeError("OpenAI /embeddings failed (401): secret details")

    monkeypatch.setattr(main, "answer_question", failing_answer)

    response = client.post("/api/ask", json={"question": "Bonjour"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not answer the question"}
    assert "secret" not in response.text
