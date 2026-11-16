"""Request and response models of the REST API (also the OpenAPI / Swagger documentation)."""

from pydantic import BaseModel, ConfigDict, Field

from app.domain import MONDRIA, Pipeline, Strategy, UploadKind

COLLECTION_PATTERN = r"^(mondria|upload-[0-9a-f]{32})$"


class AskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "Où sont les bureaux de Mondria ?"},
                {"question": "Quanto custa o suporte de TI?", "strategy": "hybrid"},
                {"question": "Where are Mondria's offices?", "strategy": "rerank", "pipeline": "langchain"},
            ]
        }
    )

    question: str = Field(min_length=1, max_length=300, description="Any language.")
    strategy: Strategy | None = Field(
        default=None, description="Retrieval strategy; the server default (rerank) when omitted."
    )
    pipeline: Pipeline = Field(
        default="native",
        description=(
            '"native" (this project\'s code) or "langchain" (same strategies with LangChain components; '
            "vector, hybrid, multi_query and rerank only)."
        ),
    )
    collection: str = Field(
        default=MONDRIA,
        pattern=COLLECTION_PATTERN,
        description=(
            '"mondria" for the demo knowledge base, or the collection returned by POST /api/documents.'
        ),
    )


class SourceDocument(BaseModel):
    id: str = Field(examples=["MON-01"])
    title: str = Field(examples=["Entreprise Mondria"])
    file: str = Field(examples=["01-entreprise-mondria.md"])
    source_urls: list[str] = Field(examples=[["https://mondria.com/"]])


class Source(BaseModel):
    id: str = Field(examples=["MON-01-04"], description="Section ID, as cited in the answer.")
    title: str = Field(examples=["Coordonnées"])
    text: str = Field(description="Full text of the section, with its heading.")
    document: SourceDocument
    score: float = Field(examples=[1.0], description="Relevance; its scale depends on the strategy.")
    cited: bool = Field(description="Whether the answer cites this section.")


class AskResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "Les bureaux de Mondria sont à Sherbrooke et à Boisbriand [MON-01-04].",
                    "strategy": "rerank",
                    "pipeline": "native",
                    "sources": [
                        {
                            "id": "MON-01-04",
                            "title": "Coordonnées",
                            "text": "## MON-01-04 — Coordonnées\n\nBureaux publiés : 36, rue Wellington Sud…",
                            "document": {
                                "id": "MON-01",
                                "title": "Entreprise Mondria",
                                "file": "01-entreprise-mondria.md",
                                "source_urls": ["https://mondria.com/"],
                            },
                            "score": 1.0,
                            "cited": True,
                        }
                    ],
                }
            ]
        }
    )

    answer: str
    sources: list[Source]
    strategy: Strategy
    pipeline: Pipeline


class UploadResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "collection": "upload-3f2b8c1d9e7a4b6c8d0e1f2a3b4c5d6e",
                    "title": "Guide du télétravail",
                    "file": "guide.md",
                    "kind": "markdown",
                    "size_bytes": 443,
                    "sections": ["Horaires", "Équipement", "Sécurité"],
                    "language": "français",
                    "summary": "Ce guide décrit les règles du télétravail.",
                    "suggested_questions": ["Combien de jours de télétravail par semaine ?"],
                    "expires_at": "2026-11-16T22:59:04+00:00",
                }
            ]
        }
    )

    collection: str = Field(description="Private ID to send in POST /api/ask; only its holder can query it.")
    title: str
    file: str
    kind: UploadKind
    size_bytes: int
    sections: list[str] = Field(
        description="Title of each indexed section, in order (DOC-01-01, DOC-01-02…)."
    )
    language: str
    summary: str
    suggested_questions: list[str]
    expires_at: str = Field(description="ISO 8601; the document is deleted after 24 hours.")


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class ErrorDetail(BaseModel):
    code: str = Field(examples=["document_not_found"])
    message: str = Field(examples=["Document not found or expired"])
    request_id: str | None = Field(examples=["5f0c2a9e1b3d4c7a8e6f0d2b4a6c8e0f"])


class ErrorResponse(BaseModel):
    detail: str | list[object] = Field(description="Same message, kept for compatibility.")
    error: ErrorDetail


def error_responses(*status_codes: int) -> dict[int | str, dict[str, object]]:
    """OpenAPI documentation of the error responses a route can return."""
    descriptions = {
        400: "Invalid document (empty, unreadable, too many sections)",
        404: "Document not found or expired",
        413: "File larger than 1 MB",
        415: "Unsupported file type",
        422: "Invalid request, or a strategy the chosen pipeline doesn't offer",
        429: "Too many requests (Vercel Firewall rate limit; not this JSON format)",
        500: "Internal server error",
    }
    return {code: {"model": ErrorResponse, "description": descriptions[code]} for code in status_codes}
