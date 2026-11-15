import json

import pytest

import rag
from db import Document, ScoredChunk

DOCUMENT = Document(id="MON-01", title="Entreprise Mondria", file="01.md", source_urls=["https://mondria.com/"])


def source(section_id: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        id=section_id, title="T", text=f"## {section_id} — T\n\nTexte.", document=DOCUMENT, score=score, cited=False
    )


def test_build_context_prefixes_each_chunk_with_its_id():
    context = rag.build_context([source("MON-01-01"), source("MON-01-04")])

    assert context.startswith("Context:\n[MON-01-01] ## MON-01-01")
    assert "\n\n[MON-01-04] ## MON-01-04" in context


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replaces retrieval and the chat call; returns the messages sent to the model."""
    sent: dict = {}

    async def fake_retrieve(question: str, top_k: int = rag.TOP_K):
        return [source("MON-01-04", 0.6), source("MON-01-02", 0.4), source("MON-02-04", 0.3)]

    def fake_chat_returning(answer: str):
        async def fake_chat(messages, json_schema=None):
            sent["messages"] = messages
            sent["json_schema"] = json_schema
            return json.dumps({"question_language": "French", "answer": answer})
        return fake_chat

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    return lambda answer: (monkeypatch.setattr(rag, "chat", fake_chat_returning(answer)), sent)[1]


@pytest.mark.anyio
async def test_answer_marks_cited_sources_with_or_without_brackets(fake_pipeline):
    fake_pipeline("Deux bureaux [MON-01-04]. Aucun tarif fixe (section MON-02-04).")

    result = await rag.answer_question("Où sont les bureaux ?")

    cited = {s["id"]: s["cited"] for s in result["sources"]}
    assert cited == {"MON-01-04": True, "MON-01-02": False, "MON-02-04": True}


@pytest.mark.anyio
async def test_answer_without_citation_marks_nothing(fake_pipeline):
    fake_pipeline("Cette information n'est pas disponible.")

    result = await rag.answer_question("Qui est le PDG ?")

    assert result["answer"] == "Cette information n'est pas disponible."
    assert not any(s["cited"] for s in result["sources"])


@pytest.mark.anyio
async def test_question_is_sent_last_with_the_structured_answer_schema(fake_pipeline):
    sent = fake_pipeline("ok")

    await rag.answer_question("Quanto custa?")

    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "system", "user"]
    assert sent["messages"][-1]["content"] == "Quanto custa?"
    assert sent["messages"][1]["content"].startswith("Context:")
    assert sent["json_schema"]["schema"]["required"] == ["question_language", "answer"]
