"""The FastAPI application, served under /api by Vercel Services (api/main.py re-exports `app`)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from mcp.server.transport_security import TransportSecuritySettings

from app.config import Settings
from app.errors import register_error_handlers
from app.logging import RequestContextMiddleware, configure_logging
from app.mcp import build_mcp_server
from app.routers import ask, documents, health
from app.runtime import open_runtime

DESCRIPTION = """
Retrieval-Augmented Generation over a French knowledge base about Mondria, and over documents
that visitors upload. Answers cite the sections they come from (`[MON-01-04]`, `[DOC-01-02]`).

- **ask**: question → answer with cited sources, six retrieval strategies
- **documents**: upload a `.md`, `.txt` or `.pdf` into a private collection (24 h)
- The same knowledge base is also an MCP server at `/api/mcp` (Streamable HTTP, not documented here).

Errors share one format: `{"detail": "...", "error": {"code", "message", "request_id"}}`.
Every response carries an `X-Request-ID` header.
"""

TAGS = [
    {"name": "ask", "description": "Questions answered from the retrieved sections."},
    {"name": "documents", "description": "Visitors' own documents, isolated and temporary."},
    {"name": "health", "description": "Liveness."},
]


def create_app(
    settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    """A fresh app: its own runtime (opened in the lifespan) and its own MCP server."""
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with open_runtime(settings, transport) as runtime, mcp_server.session_manager.run():
            app.state.runtime = runtime
            yield

    app = FastAPI(
        title="Mondria RAG API",
        version="1.0.0",
        description=DESCRIPTION,
        openapi_tags=TAGS,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    mcp_server = build_mcp_server(lambda: app.state.runtime)
    # Remote MCP endpoint. Stateless with JSON responses: on serverless, any function instance can
    # answer any request, so there is no session to keep between calls. DNS rebinding protection
    # guards local servers from malicious web pages; this one is public on purpose, so it's off.
    mcp_http_app = mcp_server.streamable_http_app(
        streamable_http_path="/api/mcp",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    register_error_handlers(app)
    app.add_middleware(RequestContextMiddleware)
    for router in (ask.router, documents.router, health.router):
        app.include_router(router)
    # Mounted last: the routes above match first, and the MCP app only serves /api/mcp.
    app.mount("/", mcp_http_app)
    return app


app = create_app()
