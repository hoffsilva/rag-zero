"""FastAPI app exposing the RAG pipeline. Served under /api by Vercel Services."""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import ScoredChunk, answer_question

logger = logging.getLogger(__name__)

app = FastAPI(title="Mondria RAG API")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300)


class AskResponse(BaseModel):
    answer: str
    sources: list[ScoredChunk]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask")
async def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Empty question")
    try:
        return AskResponse(**await answer_question(question))
    except Exception:
        logger.exception("Could not answer the question")
        raise HTTPException(status_code=500, detail="Could not answer the question")
