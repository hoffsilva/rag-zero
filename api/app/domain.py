"""Types and constants shared by the services and repositories."""

from typing import Literal, TypedDict, get_args

MONDRIA = "mondria"  # collection of the demo knowledge base
UPLOAD_TTL_HOURS = 24

Strategy = Literal["vector", "keyword", "hybrid", "mmr", "multi_query", "rerank"]
STRATEGIES: tuple[Strategy, ...] = get_args(Strategy)

# "native" is this project's own code; "langchain" runs the same strategies with LangChain components.
Pipeline = Literal["native", "langchain"]
PIPELINES: tuple[Pipeline, ...] = get_args(Pipeline)
LANGCHAIN_STRATEGIES: tuple[Strategy, ...] = ("vector", "hybrid", "multi_query", "rerank")

UploadKind = Literal["markdown", "text", "pdf"]


class Document(TypedDict):
    id: str
    title: str
    file: str
    source_urls: list[str]


class Chunk(TypedDict):
    id: str
    title: str
    text: str
    document: Document


class ScoredChunk(Chunk):
    score: float
    cited: bool


class Answer(TypedDict):
    answer: str
    sources: list[ScoredChunk]
    strategy: Strategy
    pipeline: Pipeline


class Upload(TypedDict):
    collection: str
    title: str
    file: str
    kind: UploadKind
    size_bytes: int
    sections: list[str]
    language: str
    summary: str
    suggested_questions: list[str]
    expires_at: str
