"""Golden tests: the observable behavior of the API, end to end, must not change.

The whole stack runs for real (FastAPI app, retrieval strategies, Postgres + pgvector in a
temporary schema) except OpenAI, replaced by a deterministic fake:
- embeddings: bag-of-words hashed into 1536 dimensions, normalized, so similar texts get close
  vectors and rankings are meaningful and stable;
- chat: a fixed reply per kind of call (answer, query rewrites, reranking, document description).

The expected results live in tests/golden/api.json, generated from the code before the
refactoring. Regenerate only on purpose: UPDATE_GOLDEN=1 pytest tests/test_golden.py
"""

import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

GOLDEN_FILE = Path(__file__).parent / "golden" / "api.json"
STRATEGIES = ["vector", "keyword", "hybrid", "mmr", "multi_query", "rerank"]
QUESTIONS = [
    "Où sont les bureaux de Mondria ?",
    "Mondria travaille-t-elle avec des OBNL ?",
    "How much do IT services cost?",
]
UPLOAD = (
    "# Guide du télétravail\n\n## Horaires\nLes employés travaillent à distance trois jours par semaine.\n\n"
    "## Équipement\nLes frais d'internet sont remboursés jusqu'à 30 $ par mois.\n\n"
    "## Sécurité\nLe VPN est obligatoire pour accéder aux systèmes internes.\n"
)

pytestmark = pytest.mark.db


def fake_embedding(text: str) -> list[float]:
    vector = [0.0] * 1536
    for word in re.findall(r"\w+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 1536] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def fake_chat_content(body: dict) -> str:
    messages = body["messages"]
    schema = body.get("response_format", {}).get("json_schema", {}).get("name")
    if schema == "rag_answer":
        context = messages[1]["content"]
        first_id = re.search(r"\[([A-Z]+-\d+-\d+)\]", context)
        cited = f" [{first_id.group(1)}]" if first_id else ""
        return json.dumps({"question_language": "French", "answer": f"Réponse de test{cited}."})
    if schema == "query_variants":
        return json.dumps({"queries": ["bureaux adresse Mondria", "secteurs organismes", "tarifs services TI"]})
    if schema == "reranking":
        ids = re.findall(r"^\[([A-Z]+-\d+-\d+)\]", messages[-1]["content"], re.MULTILINE)
        return json.dumps({"ranked_ids": sorted(ids)})  # deterministic, different from the input order
    if body.get("response_format", {}).get("type") == "json_object":
        return json.dumps({"language": "français", "summary": "Un guide.", "suggested_questions": ["Combien de jours ?"]})
    raise AssertionError(f"unexpected chat call: {schema}")


async def fake_openai(path: str, body: dict) -> dict:
    if path == "/embeddings":
        return {"data": [{"index": i, "embedding": fake_embedding(text)} for i, text in enumerate(body["input"])]}
    if path == "/chat/completions":
        return {"choices": [{"finish_reason": "stop", "message": {"content": fake_chat_content(body), "refusal": None}}]}
    raise AssertionError(f"unexpected OpenAI call: {path}")


def install_fake_openai(monkeypatch) -> None:
    """The only line that follows the refactoring: where the OpenAI HTTP boundary lives."""
    import openai_client

    monkeypatch.setattr(openai_client, "_post", fake_openai)


def reset_app_state(monkeypatch) -> None:
    import rag

    monkeypatch.setattr(rag, "_ready_model", None)


@pytest.fixture
def client(monkeypatch):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    schema = f"golden_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    separator = "&" if "?" in url else "?"
    monkeypatch.setenv("DATABASE_URL", f"{url}{separator}options=-csearch_path%3D{schema}%2Cpublic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "fake-chat")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "rerank")
    install_fake_openai(monkeypatch)
    reset_app_state(monkeypatch)
    import main

    try:
        with TestClient(main.app) as test_client:
            yield test_client
    finally:
        with psycopg.connect(url, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


def ask_summary(response) -> dict:
    assert response.status_code == 200, response.text
    body = response.json()
    return {
        "strategy": body["strategy"],
        "answer": body["answer"],
        "sources": [
            {"id": s["id"], "score": round(s["score"], 6), "cited": s["cited"], "document": s["document"]["id"]}
            for s in body["sources"]
        ],
    }


def collect(client: TestClient) -> dict:
    results: dict = {"mondria": {}, "upload": {}}
    for strategy in STRATEGIES:
        results["mondria"][strategy] = [
            ask_summary(client.post("/api/ask", json={"question": q, "strategy": strategy})) for q in QUESTIONS
        ]
    results["mondria"]["default"] = ask_summary(client.post("/api/ask", json={"question": QUESTIONS[0]}))

    response = client.post("/api/documents", files={"file": ("guide.md", UPLOAD.encode(), "text/markdown")})
    assert response.status_code == 200, response.text
    upload = response.json()
    collection = upload.pop("collection")
    upload.pop("expires_at")
    results["upload"]["document"] = upload
    results["upload"]["answers"] = {
        strategy: ask_summary(
            client.post("/api/ask", json={"question": "Le VPN est-il obligatoire ?", "collection": collection, "strategy": strategy})
        )
        for strategy in STRATEGIES
    }
    return results


def test_api_behavior_matches_the_golden_snapshot(client):
    results = collect(client)

    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN_FILE.parent.mkdir(exist_ok=True)
        GOLDEN_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pytest.skip("golden snapshot regenerated")

    assert results == json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
