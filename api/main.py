"""FastAPI app exposing the RAG pipeline. Served under /api by Vercel Services.

- POST /api/ask: question -> answer with cited sources, in the Mondria base or an uploaded document
- POST /api/documents: upload a document into a private collection that expires after 24 h
- GET /api/documents/{collection}: that upload's metadata (summary, sections, suggested questions)
- /api/mcp: the Mondria knowledge base as a remote MCP server (Streamable HTTP)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

import documents
import mcp_server
from db import MONDRIA, Upload
from rag import ScoredChunk, answer_question
from retrieval import Strategy

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
    strategy: Strategy | None = None  # RETRIEVAL_STRATEGY when omitted
    # "mondria" (default) or the collection returned by POST /api/documents.
    collection: str = Field(default=MONDRIA, pattern=r"^(mondria|upload-[0-9a-f]{32})$")


class AskResponse(BaseModel):
    answer: str
    sources: list[ScoredChunk]
    strategy: Strategy


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask")
async def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Empty question")
    if request.collection != MONDRIA and await documents.get_upload(request.collection) is None:
        raise HTTPException(status_code=404, detail="Document not found or expired")
    try:
        return AskResponse(**await answer_question(question, request.strategy, request.collection))
    except Exception:
        logger.exception("Could not answer the question")
        raise HTTPException(status_code=500, detail="Could not answer the question")


@app.post("/api/documents")
async def upload_document(file: UploadFile) -> Upload:
    data = await file.read(documents.MAX_BYTES + 1)  # never read more than the limit
    try:
        return await documents.ingest(file.filename or "document.txt", data)
    except documents.DocumentError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)
    except Exception:
        logger.exception("Could not ingest the document")
        raise HTTPException(status_code=500, detail="Could not process the document")


@app.get("/api/documents/{collection}")
async def get_document(collection: str) -> Upload:
    upload = await documents.get_upload(collection) if documents.is_upload_collection(collection) else None
    if upload is None:
        raise HTTPException(status_code=404, detail="Document not found or expired")
    return upload


# Mounted last: the routes above match first, and the MCP app only serves /api/mcp.
app.mount("/", mcp_http_app)
