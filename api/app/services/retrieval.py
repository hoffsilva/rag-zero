"""Retrieval strategies. Each one turns a question into the top-k chunks sent to the LLM.

| strategy     | idea                                                                  | extra cost      |
|--------------|-----------------------------------------------------------------------|-----------------|
| vector       | cosine similarity between the question and chunk embeddings           | -               |
| keyword      | Postgres full-text search (French stemming), no embedding at all       | -               |
| hybrid       | vector + keyword result lists fused with Reciprocal Rank Fusion         | -               |
| mmr          | vector candidates re-picked for relevance *and* diversity              | -               |
| multi_query  | the LLM rewrites the question in French 3 ways; vector search; RRF      | 1 chat call     |
| rerank       | vector candidates reordered by the LLM by relevance                    | 1 chat call     |

The score meaning depends on the strategy: cosine similarity (vector, mmr), ts_rank (keyword),
RRF score (hybrid, multi_query) or a position score (rerank). Compare ranks, not scores.
"""

import json
from typing import Any

from app.clients.openai import JsonObject, OpenAIClient
from app.domain import MONDRIA, ScoredChunk, Strategy
from app.repositories import chunks as chunk_repository
from app.repositories.database import Connection

CANDIDATES = 20  # how many chunks each first-stage search returns before fusion / re-picking
RERANK_CANDIDATES = 10
RRF_K = 60  # standard constant from the RRF paper; damps the weight of the very first ranks
MMR_LAMBDA = 0.7  # 1 = pure relevance, 0 = pure diversity
MULTI_QUERY_VARIANTS = 3


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]], top_k: int, k: int = RRF_K
) -> list[ScoredChunk]:
    """Merges ranked lists: each chunk scores the sum of 1 / (k + rank) over the lists it appears
    in. Only ranks matter, so lists with incomparable scores (cosine vs ts_rank) fuse fairly."""
    scores: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk["id"]] = scores.get(chunk["id"], 0.0) + 1 / (k + rank)
            chunks.setdefault(chunk["id"], chunk)
    ordered = sorted(scores, key=lambda section_id: (-scores[section_id], section_id))
    return [
        ScoredChunk(**{**chunks[section_id], "score": scores[section_id]}) for section_id in ordered[:top_k]
    ]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # cosine similarity: embeddings have length 1


def maximal_marginal_relevance(
    query_embedding: list[float],
    candidates: list[tuple[ScoredChunk, list[float]]],
    top_k: int,
    lambda_: float = MMR_LAMBDA,
) -> list[ScoredChunk]:
    """Greedily picks the candidate that maximizes
    lambda * similarity(question) - (1 - lambda) * max similarity(already picked),
    so a chunk that repeats one already picked loses to a slightly less relevant, new one."""
    remaining = list(candidates)
    picked: list[tuple[ScoredChunk, list[float]]] = []
    while remaining and len(picked) < top_k:

        def mmr_score(candidate: tuple[ScoredChunk, list[float]]) -> float:
            redundancy = max((dot(candidate[1], other[1]) for other in picked), default=0.0)
            return lambda_ * dot(query_embedding, candidate[1]) - (1 - lambda_) * redundancy

        best = max(remaining, key=mmr_score)
        picked.append(best)
        remaining.remove(best)
    return [chunk for chunk, _ in picked]


MULTI_QUERY_SCHEMA: JsonObject = {
    "name": "query_variants",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
        "required": ["queries"],
        "additionalProperties": False,
    },
}

RERANK_SCHEMA: JsonObject = {
    "name": "reranking",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"ranked_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["ranked_ids"],
        "additionalProperties": False,
    },
}


MONDRIA_SUBJECT = (
    "a knowledge base about the company Mondria (IT services, cybersecurity, AI), written in French"
)


async def rewrite_question(openai: OpenAIClient, question: str, subject: str = MONDRIA_SUBJECT) -> list[str]:
    """Asks the LLM for search queries in the documents' language: for the Mondria base (French),
    a Portuguese or English question gets French wording, and an ambiguous one gets several angles."""
    raw = await openai.chat(
        [
            {
                "role": "system",
                "content": (
                    f"Rewrite the user's question as {MULTI_QUERY_VARIANTS} different search queries "
                    f"for {subject}. Write them in the language of the documents. "
                    "Keep the meaning; vary the wording and the angle."
                ),
            },
            {"role": "user", "content": question},
        ],
        json_schema=MULTI_QUERY_SCHEMA,
    )
    queries: list[Any] = json.loads(raw)["queries"]
    return [str(query) for query in queries if str(query).strip()][:MULTI_QUERY_VARIANTS]


async def rerank_with_llm(
    openai: OpenAIClient, question: str, candidates: list[ScoredChunk], top_k: int
) -> list[ScoredChunk]:
    """Asks the LLM to order the candidate sections by how well they answer the question.
    Unknown IDs are ignored and candidates the model left out keep their original order at the end."""
    listing = "\n\n".join(f"[{chunk['id']}] {chunk['text']}" for chunk in candidates)
    raw = await openai.chat(
        [
            {
                "role": "system",
                "content": (
                    "You rank sections of a knowledge base by how useful they are to answer a question. "
                    "Return the section IDs, most useful first."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nSections:\n{listing}"},
        ],
        json_schema=RERANK_SCHEMA,
    )
    by_id = {chunk["id"]: chunk for chunk in candidates}
    ranked_ids: list[Any] = json.loads(raw)["ranked_ids"]
    ordered_ids = [section_id for section_id in dict.fromkeys(map(str, ranked_ids)) if section_id in by_id]
    ordered_ids += [chunk["id"] for chunk in candidates if chunk["id"] not in ordered_ids]
    total = len(ordered_ids)
    return [
        ScoredChunk(**{**by_id[section_id], "score": (total - position) / total})
        for position, section_id in enumerate(ordered_ids[:top_k])
    ]


async def run(
    strategy: Strategy,
    openai: OpenAIClient,
    connection: Connection,
    question: str,
    top_k: int,
    collection: str = MONDRIA,
    subject: str = MONDRIA_SUBJECT,
) -> list[ScoredChunk]:
    model = openai.embedding_model
    if strategy == "keyword":
        return await chunk_repository.keyword_search(connection, question, model, top_k, collection)

    if strategy == "multi_query":
        variants = await rewrite_question(openai, question, subject)
        embeddings = await openai.embed([question, *variants], model)  # one batch call
        rankings = [
            await chunk_repository.search(connection, embedding, model, CANDIDATES, collection)
            for embedding in embeddings
        ]
        return reciprocal_rank_fusion(rankings, top_k)

    [query_embedding] = await openai.embed([question], model)

    if strategy == "vector":
        return await chunk_repository.search(connection, query_embedding, model, top_k, collection)

    if strategy == "rerank":
        # Candidates come from the vector search, not the hybrid one: French full-text search misses
        # Portuguese and English questions, which kept the right section out of the pool (measured
        # in eval/: hybrid candidates capped rerank at 87% hit rate@5).
        candidates = await chunk_repository.search(
            connection, query_embedding, model, RERANK_CANDIDATES, collection
        )
        return await rerank_with_llm(openai, question, candidates, top_k)

    if strategy == "mmr":
        pool = await chunk_repository.search_with_embeddings(
            connection, query_embedding, model, CANDIDATES, collection
        )
        return maximal_marginal_relevance(query_embedding, pool, top_k)

    if strategy == "hybrid":
        rankings = [
            await chunk_repository.search(connection, query_embedding, model, CANDIDATES, collection),
            await chunk_repository.keyword_search(connection, question, model, CANDIDATES, collection),
        ]
        return reciprocal_rank_fusion(rankings, top_k)

    raise ValueError(f"Unknown strategy {strategy!r}")
