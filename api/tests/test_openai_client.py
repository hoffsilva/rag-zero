import pytest

import openai_client


@pytest.fixture
def sent(monkeypatch):
    """Captures the request body instead of calling OpenAI; embeddings come back out of order."""
    requests: list[dict] = []

    async def fake_post(path: str, body: dict) -> dict:
        requests.append(body)
        return {"data": [{"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]}]}

    monkeypatch.setattr(openai_client, "_post", fake_post)
    return requests


@pytest.mark.anyio
@pytest.mark.parametrize("model", ["text-embedding-3-small", "text-embedding-3-large"])
async def test_text_embedding_3_models_are_asked_for_the_column_size(sent, model):
    await openai_client.embed(["a", "b"], model)

    assert sent[0]["dimensions"] == 1536


@pytest.mark.anyio
async def test_ada_002_gets_no_dimensions_parameter(sent):
    # text-embedding-ada-002 is always 1536 and rejects the parameter.
    await openai_client.embed(["a"], "text-embedding-ada-002")

    assert "dimensions" not in sent[0]


@pytest.mark.anyio
async def test_embeddings_are_returned_in_input_order(sent):
    embeddings = await openai_client.embed(["a", "b"], "text-embedding-3-small")

    assert embeddings == [[0.1], [0.2]]
