"""The same RAG, orchestrated with LangChain components, to compare with the native pipeline.

Only the orchestration changes: both pipelines read the same `chunks` table (through the
repositories, wrapped as LangChain retrievers), use the same models and the same prompts, and
return the same Answer. `langchain-postgres` is not used: it needs numpy, an older pgvector and
its own tables.

| strategy     | LangChain component                                                         |
|--------------|-----------------------------------------------------------------------------|
| vector       | PgVectorRetriever (custom BaseRetriever over chunks.search)                  |
| hybrid       | EnsembleRetriever(vector, keyword): weighted Reciprocal Rank Fusion          |
| multi_query  | MultiQueryRetriever: LLM query variants, unique union of their results        |
| rerank       | ContextualCompressionRetriever + LLMListwiseRerank over 10 vector candidates |

`keyword` would be the same SQL as the native one, and LangChain's MMR needs numpy and a
vector store, so neither is offered here.
"""

from collections.abc import Sequence
from typing import Any

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
    Callbacks,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import ModelRefusalError
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from app.domain import LANGCHAIN_STRATEGIES, MONDRIA, Answer, ScoredChunk, Strategy
from app.errors import StrategyNotAvailableError
from app.repositories import chunks as chunk_repository
from app.repositories.database import Connection
from app.runtime import Runtime
from app.services import rag
from app.services.indexing import open_connection
from app.services.retrieval import CANDIDATES, MULTI_QUERY_VARIANTS, RERANK_CANDIDATES, RRF_K

CHUNK_KEY = "chunk"  # Document.metadata key holding the original ScoredChunk


def to_document(chunk: ScoredChunk) -> Document:
    return Document(
        page_content=chunk["text"], id=chunk["id"], metadata={"id": chunk["id"], CHUNK_KEY: chunk}
    )


def to_chunks(documents: Sequence[Document], top_k: int, keep_scores: bool) -> list[ScoredChunk]:
    """Back to ScoredChunk. Fusion and reranking components don't expose scores, so those
    results get a position score (1 for the first, then decreasing), like the native rerank."""
    # Deduplicated by section ID: MultiQueryRetriever compares whole Documents, and the same
    # section found by two queries carries two different scores.
    unique = {document.metadata["id"]: document for document in reversed(documents)}
    picked = [unique[section_id] for section_id in dict.fromkeys(d.metadata["id"] for d in documents)][:top_k]
    total = len(picked)
    chunks: list[ScoredChunk] = []
    for position, document in enumerate(picked):
        chunk: ScoredChunk = document.metadata[CHUNK_KEY]
        score = chunk["score"] if keep_scores else (total - position) / total
        chunks.append(ScoredChunk(**{**chunk, "score": score, "cited": False}))
    return chunks


class _PgRetriever(BaseRetriever):
    """Base for retrievers bound to one request's connection. Async only."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: SkipValidation[Connection]  # a live psycopg connection; nothing to validate
    embedding_model: str
    collection: str
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError("Use ainvoke: the database connection is async")


class PgVectorRetriever(_PgRetriever):
    """Embeds the query with the LangChain embeddings, then runs the pgvector cosine search."""

    embeddings: Embeddings

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_embedding = await self.embeddings.aembed_query(query)
        rows = await chunk_repository.search(
            self.connection, query_embedding, self.embedding_model, self.k, self.collection
        )
        return [to_document(chunk) for chunk in rows]


class PgKeywordRetriever(_PgRetriever):
    """Postgres French full-text search (no embedding)."""

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        rows = await chunk_repository.keyword_search(
            self.connection, query, self.embedding_model, self.k, self.collection
        )
        return [to_document(chunk) for chunk in rows]


class AsyncListwiseRerank(LLMListwiseRerank):
    """LLMListwiseRerank only implements the sync path (the async one runs it in a thread);
    this awaits the same reranking chain instead."""

    async def acompress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        results: Sequence[Document] = await self.reranker.ainvoke(
            {"documents": documents, "query": query}, config={"callbacks": callbacks}
        )
        return results[: self.top_n]


# Same instruction as the native multi_query (retrieval.rewrite_question): variants in the
# documents' language. LangChain's default prompt keeps the question's language, which misses
# French documents for Portuguese and English questions.
MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    f"Rewrite the user's question as {MULTI_QUERY_VARIANTS} different search queries for {{subject}}. "
    "Write them in the language of the documents. Keep the meaning; vary the wording and the angle. "
    "Return one query per line, nothing else.\n\nQuestion: {question}"
)


class RagAnswer(BaseModel):
    """Structured answer; naming the question's language first keeps the answer in it."""

    question_language: str = Field(description="Language the user's question is written in.")
    answer: str = Field(
        description="The answer, written in question_language, citing section IDs like [MON-01-04]."
    )


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    # Values are inserted as text, so braces in the context are never read as variables.
    [("system", "{system}"), ("system", "{context}"), ("human", "{question}")]
)


def not_available(strategy: Strategy) -> StrategyNotAvailableError:
    return StrategyNotAvailableError(
        f"Strategy {strategy!r} is not available in the LangChain pipeline. "
        f"Available: {', '.join(LANGCHAIN_STRATEGIES)}."
    )


def build_retriever(
    runtime: Runtime, connection: Connection, strategy: Strategy, collection: str
) -> tuple[BaseRetriever, bool]:
    """The LangChain retriever for a strategy, and whether its results keep their own scores."""
    models = runtime.langchain
    common: dict[str, Any] = {
        "connection": connection,
        "embedding_model": runtime.openai.embedding_model,
        "collection": collection,
    }

    def vector(k: int) -> PgVectorRetriever:
        return PgVectorRetriever(embeddings=models.embeddings, k=k, **common)

    if strategy == "vector":
        return vector(rag.TOP_K), True
    if strategy == "hybrid":
        keyword = PgKeywordRetriever(k=CANDIDATES, **common)
        return EnsembleRetriever(
            retrievers=[vector(CANDIDATES), keyword], weights=[0.5, 0.5], c=RRF_K, id_key="id"
        ), False
    if strategy == "multi_query":
        prompt = MULTI_QUERY_PROMPT.partial(subject=rag.subject_for(collection))
        return MultiQueryRetriever.from_llm(
            retriever=vector(rag.TOP_K), llm=models.chat, prompt=prompt, include_original=True
        ), False
    if strategy == "rerank":
        compressor = AsyncListwiseRerank.from_llm(models.chat, top_n=rag.TOP_K)
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vector(RERANK_CANDIDATES)
        ), False
    raise not_available(strategy)


async def retrieve(
    runtime: Runtime, question: str, strategy: Strategy, collection: str = MONDRIA
) -> list[ScoredChunk]:
    async with await open_connection(runtime) as connection:
        retriever, keep_scores = build_retriever(runtime, connection, strategy, collection)
        documents = await retriever.ainvoke(question)
    return to_chunks(documents, rag.TOP_K, keep_scores)


async def answer_question(
    runtime: Runtime, question: str, strategy: Strategy | None = None, collection: str = MONDRIA
) -> Answer:
    strategy = strategy or runtime.settings.retrieval_strategy
    if strategy not in LANGCHAIN_STRATEGIES:
        raise not_available(strategy)
    sources = await retrieve(runtime, question, strategy, collection)

    chain = ANSWER_PROMPT | runtime.langchain.chat.with_structured_output(
        RagAnswer, method="json_schema", strict=True
    )
    try:
        reply = await chain.ainvoke(
            {
                "system": rag.system_prompt_for(collection),
                "context": rag.build_context(sources),
                "question": question,
            }
        )
    except ModelRefusalError as refusal:
        return Answer(answer=str(refusal), sources=sources, strategy=strategy, pipeline="langchain")
    answer = RagAnswer.model_validate(reply).answer
    rag.mark_cited(answer, sources)
    return Answer(answer=answer, sources=sources, strategy=strategy, pipeline="langchain")
