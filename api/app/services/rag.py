"""Retrieval-Augmented Generation: retrieve the sections, build the prompt, generate the answer."""

import json
import re
from typing import Any

from app.clients.openai import ChatRefusal, JsonObject
from app.domain import MONDRIA, Answer, ScoredChunk, Strategy
from app.runtime import Runtime
from app.services import retrieval
from app.services.indexing import open_connection

TOP_K = 4

# A section ID as a whole token: "[MON-01-04]" or "MON-01-04", but not inside "MON-01-040".
CITED_ID_PATTERN = re.compile(r"(?<![\w-])[A-Z]+-\d+-\d+(?![\w-])")

SYSTEM_PROMPT = """You answer questions about the company Mondria using a knowledge base written in French.
Answer only from the provided context.
Cite the IDs of the sections you used in square brackets, for example [MON-01-04].
If the information is not in the context, say clearly that it is not available. Never make anything up.
Always answer in the language of the user's question (for example Portuguese, English or French), \
even though the context is in French. This also applies when the information is not available. Be concise."""

UPLOAD_SYSTEM_PROMPT = """You answer questions about a document the user uploaded.
Answer only from the provided context, which contains sections of that document.
Cite the IDs of the sections you used in square brackets, for example [DOC-01-03].
If the information is not in the context, say clearly that it is not in the document. Never make anything up.
Always answer in the language of the user's question, whatever the document's language. Be concise."""

UPLOAD_SUBJECT = "a document the user uploaded"

ANSWER_SCHEMA: JsonObject = {
    "name": "rag_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "question_language": {
                "type": "string",
                "description": "Language the user's question is written in.",
            },
            "answer": {
                "type": "string",
                "description": (
                    "The answer, written in question_language, citing section IDs like [MON-01-04]."
                ),
            },
        },
        "required": ["question_language", "answer"],
        "additionalProperties": False,
    },
}


def build_context(sources: list[ScoredChunk]) -> str:
    """Augmented step: the retrieved chunks become the context of the prompt."""
    return "Context:\n" + "\n\n".join(f"[{chunk['id']}] {chunk['text']}" for chunk in sources)


async def retrieve(
    runtime: Runtime,
    question: str,
    top_k: int = TOP_K,
    strategy: Strategy | None = None,
    collection: str = MONDRIA,
) -> list[ScoredChunk]:
    """Retrieval step with the given strategy (RETRIEVAL_STRATEGY by default; see retrieval.py),
    in the Mondria base or in an uploaded document's collection. The question is embedded with
    the same model as the chunks."""
    subject = retrieval.MONDRIA_SUBJECT if collection == MONDRIA else UPLOAD_SUBJECT
    async with await open_connection(runtime) as connection:
        return await retrieval.run(
            strategy or runtime.settings.retrieval_strategy,
            runtime.openai,
            connection,
            question,
            top_k,
            collection=collection,
            subject=subject,
        )


async def answer_question(
    runtime: Runtime, question: str, strategy: Strategy | None = None, collection: str = MONDRIA
) -> Answer:
    strategy = strategy or runtime.settings.retrieval_strategy
    sources = await retrieve(runtime, question, strategy=strategy, collection=collection)
    system_prompt = SYSTEM_PROMPT if collection == MONDRIA else UPLOAD_SYSTEM_PROMPT

    # Generation. With a French context, the model tends to answer in French whatever the
    # question's language. Making it name the question's language first, in a structured
    # response, keeps the answer in that language.
    try:
        raw = await runtime.openai.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": build_context(sources)},
                {"role": "user", "content": question},
            ],
            json_schema=ANSWER_SCHEMA,
        )
    except ChatRefusal as refusal:
        # A refusal is an answer, not a server error: show it, with nothing cited.
        return Answer(answer=str(refusal), sources=sources, strategy=strategy)
    reply: dict[str, Any] = json.loads(raw)
    answer = str(reply["answer"])

    # Sources the answer actually cites, as opposed to the ones that were only retrieved.
    # IDs are compared as whole tokens; the model sometimes drops the brackets.
    cited_ids = set(CITED_ID_PATTERN.findall(answer))
    for source in sources:
        source["cited"] = source["id"] in cited_ids

    return Answer(answer=answer, sources=sources, strategy=strategy)
