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


def chat_response(content: str | None, finish_reason: str = "stop", refusal: str | None = None) -> dict:
    return {"choices": [{"finish_reason": finish_reason, "message": {"content": content, "refusal": refusal}}]}


@pytest.fixture
def chat_returns(monkeypatch):
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    requests: list[dict] = []

    def set_response(response: dict) -> list[dict]:
        async def fake_post(path: str, body: dict) -> dict:
            requests.append(body)
            return response

        monkeypatch.setattr(openai_client, "_post", fake_post)
        return requests

    return set_response


@pytest.mark.anyio
async def test_chat_caps_the_answer_length(chat_returns):
    requests = chat_returns(chat_response('{"answer": "ok"}'))

    assert await openai_client.chat([{"role": "user", "content": "q"}]) == '{"answer": "ok"}'
    assert requests[0]["max_completion_tokens"] == openai_client.MAX_ANSWER_TOKENS


@pytest.mark.anyio
async def test_chat_raises_a_refusal_instead_of_returning_empty_content(chat_returns):
    chat_returns(chat_response(None, refusal="I can't help with that."))

    with pytest.raises(openai_client.ChatRefusal, match="can't help"):
        await openai_client.chat([{"role": "user", "content": "q"}])


@pytest.mark.anyio
async def test_chat_rejects_a_truncated_answer(chat_returns):
    chat_returns(chat_response('{"answer": "cut in the mid', finish_reason="length"))

    with pytest.raises(RuntimeError, match="length"):
        await openai_client.chat([{"role": "user", "content": "q"}])


@pytest.mark.anyio
async def test_chat_retries_output_with_nul_characters(monkeypatch):
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    replies = iter([chat_response('{"answer": "employ\u0000s"}'), chat_response('{"answer": "employés"}')])

    async def fake_post(path: str, body: dict) -> dict:
        return next(replies)

    monkeypatch.setattr(openai_client, "_post", fake_post)

    assert await openai_client.chat([{"role": "user", "content": "q"}]) == '{"answer": "employés"}'


@pytest.mark.anyio
async def test_chat_strips_nul_characters_when_every_attempt_has_them(monkeypatch):
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    calls = []

    async def fake_post(path: str, body: dict) -> dict:
        calls.append(body)
        return chat_response('{"answer": "a\u0000b"}')

    monkeypatch.setattr(openai_client, "_post", fake_post)

    assert await openai_client.chat([{"role": "user", "content": "q"}]) == '{"answer": "ab"}'
    assert len(calls) == openai_client.CHAT_ATTEMPTS
