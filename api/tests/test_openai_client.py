from typing import Any

import pytest
from conftest import openai_transport

from app.clients.openai import CHAT_ATTEMPTS, MAX_ANSWER_TOKENS, ChatRefusal, OpenAIClient, OpenAIError
from app.config import ConfigurationError, Settings


def chat_response(
    content: str | None, finish_reason: str = "stop", refusal: str | None = None
) -> dict[str, Any]:
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content, "refusal": refusal}}]
    }


class Recorder:
    """An OpenAI fake: records request bodies and replays the given responses in order
    (the last one repeats)."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


EMBEDDINGS_OUT_OF_ORDER = {"data": [{"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]}]}


@pytest.mark.anyio
@pytest.mark.parametrize("model", ["text-embedding-3-small", "text-embedding-3-large"])
async def test_text_embedding_3_models_are_asked_for_the_column_size(settings, model):
    recorder = Recorder(EMBEDDINGS_OUT_OF_ORDER)
    async with OpenAIClient(settings, openai_transport(recorder)) as client:
        await client.embed(["a", "b"], model)

    assert recorder.bodies[0]["dimensions"] == 1536


@pytest.mark.anyio
async def test_ada_002_gets_no_dimensions_parameter(settings):
    # text-embedding-ada-002 is always 1536 and rejects the parameter.
    recorder = Recorder(EMBEDDINGS_OUT_OF_ORDER)
    async with OpenAIClient(settings, openai_transport(recorder)) as client:
        await client.embed(["a"], "text-embedding-ada-002")

    assert "dimensions" not in recorder.bodies[0]


@pytest.mark.anyio
async def test_embeddings_are_returned_in_input_order(settings):
    async with OpenAIClient(settings, openai_transport(Recorder(EMBEDDINGS_OUT_OF_ORDER))) as client:
        assert await client.embed(["a", "b"]) == [[0.1], [0.2]]


@pytest.mark.anyio
async def test_chat_caps_the_answer_length(settings):
    recorder = Recorder(chat_response('{"answer": "ok"}'))
    async with OpenAIClient(settings, openai_transport(recorder)) as client:
        assert await client.chat([{"role": "user", "content": "q"}]) == '{"answer": "ok"}'

    assert recorder.bodies[0]["max_completion_tokens"] == MAX_ANSWER_TOKENS
    assert recorder.bodies[0]["model"] == "fake-chat"


@pytest.mark.anyio
async def test_chat_raises_a_refusal_instead_of_returning_empty_content(settings):
    recorder = Recorder(chat_response(None, refusal="I can't help with that."))
    async with OpenAIClient(settings, openai_transport(recorder)) as client:
        with pytest.raises(ChatRefusal, match="can't help"):
            await client.chat([{"role": "user", "content": "q"}])


@pytest.mark.anyio
async def test_chat_rejects_a_truncated_answer(settings):
    recorder = Recorder(chat_response('{"answer": "cut in the mid', finish_reason="length"))
    async with OpenAIClient(settings, openai_transport(recorder)) as client:
        with pytest.raises(OpenAIError, match="length"):
            await client.chat([{"role": "user", "content": "q"}])


@pytest.mark.anyio
async def test_chat_retries_output_with_nul_characters(settings):
    recorder = Recorder(chat_response('{"answer": "employ\u0000s"}'), chat_response('{"answer": "employés"}'))
    async with OpenAIClient(settings, openai_transport(recorder)) as client:
        assert await client.chat([{"role": "user", "content": "q"}]) == '{"answer": "employés"}'


@pytest.mark.anyio
async def test_chat_strips_nul_characters_when_every_attempt_has_them(settings):
    recorder = Recorder(chat_response('{"answer": "a\u0000b"}'))
    async with OpenAIClient(settings, openai_transport(recorder)) as client:
        assert await client.chat([{"role": "user", "content": "q"}]) == '{"answer": "ab"}'

    assert len(recorder.bodies) == CHAT_ATTEMPTS


@pytest.mark.anyio
async def test_a_missing_key_fails_before_any_request():
    no_key = Settings(_env_file=None, openai_chat_model="m")
    async with OpenAIClient(no_key) as client:
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            await client.embed(["a"])


@pytest.mark.anyio
async def test_the_client_must_be_opened():
    with pytest.raises(RuntimeError, match="async with"):
        await OpenAIClient(Settings(_env_file=None)).embed(["a"])
