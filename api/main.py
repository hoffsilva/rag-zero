"""FastAPI app exposing the RAG pipeline. Served under /api by Vercel Services.

- POST /api/ask: question -> answer with cited sources (used by the web app)
- /api/mcp: the same knowledge base as a remote MCP server (Streamable HTTP)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

import mcp_server
from rag import ScoredChunk, answer_question

logger = logging.getLogger(__name__)

# Remote MCP endpoint. Stateless with JSON responses: on serverless, any function instance can
# answer any request, so there is no session to keep between calls. DNS rebinding protection
# guards local servers from malicious web pages; this one is public on purpose, so it's off.
mcp_http_app = mcp_server.server.streamable_http_app(
    streamable_http_path="/api/mcp",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_server.server.session_manager.run():
        yield


app = FastAPI(title="Mondria RAG API", lifespan=lifespan)


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


# Mounted last: the routes above match first, and the MCP app only serves /api/mcp.
app.mount("/", mcp_http_app)
