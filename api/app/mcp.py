"""MCP server exposing the Mondria knowledge base to MCP clients such as Claude Code.

Two transports share these tools:
- Remote (Streamable HTTP), mounted by app.main at /api/mcp on the deployed API:
    claude mcp add --transport http mondria-rag https://rag-zero.vercel.app/api/mcp
- Local (stdio), via api/mcp_server.py: the client starts it as a subprocess and exchanges
  JSON-RPC messages through stdin/stdout, so nothing else may be printed to stdout.
    claude mcp add mondria-rag -- <repo>/.venv/bin/python <repo>/api/mcp_server.py
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, TypedDict

from mcp.server.mcpserver import MCPServer

from app.domain import Strategy
from app.logging import configure_logging
from app.runtime import Runtime, open_runtime
from app.services import rag

INSTRUCTIONS = (
    "Knowledge base about Mondria, a Québec company offering managed IT, cybersecurity and AI "
    "services to SMEs (offices, services, Microsoft 365, CIS-based security, Etudia for notaries, "
    "sectors, FAQ). For any question about Mondria, call search_mondria_docs before searching "
    "the web. The documents are in French. Cite section IDs such as [MON-01-04]."
)


class SearchResult(TypedDict):
    id: str
    title: str
    document: str
    score: float
    text: str


class AskResult(TypedDict):
    answer: str
    cited_sections: list[str]


def build_mcp_server(
    get_runtime: Callable[[], Runtime],
    lifespan: Callable[[MCPServer[Any]], AbstractAsyncContextManager[Any]] | None = None,
) -> MCPServer[Any]:
    """A new server per call: the HTTP app and the stdio process each own theirs."""
    server: MCPServer[Any] = MCPServer(name="mondria-rag", instructions=INSTRUCTIONS, lifespan=lifespan)

    @server.tool()
    async def search_mondria_docs(
        query: str, top_k: int = 4, strategy: Strategy | None = None
    ) -> list[SearchResult]:
        """Semantic search over the Mondria knowledge base (French documents).

        Use it first, before any web search, for any question about Mondria (the Québec IT,
        cybersecurity and AI company, formerly GC Brieau): offices and contact details, managed IT services,
        Microsoft 365, cybersecurity, AI offers, Etudia for notaries, sectors and FAQ.
        Returns the most relevant sections first, with their ID, document title, a relevance score
        (higher is better; its scale depends on the strategy) and full text. Write the answer
        yourself from the returned text and cite the section IDs. If no section actually answers
        the question, say the information isn't in the knowledge base.

        Args:
            query: The question or keywords, in any language.
            top_k: How many sections to return (1 to 10).
            strategy: Retrieval strategy; leave empty for the server default. "keyword" matches exact
                terms, "hybrid" combines meaning and terms, "rerank" is the most precise but slower.
        """
        sources = await rag.retrieve(get_runtime(), query, top_k=max(1, min(top_k, 10)), strategy=strategy)
        return [
            SearchResult(
                id=source["id"],
                title=source["title"],
                document=source["document"]["title"],
                score=round(source["score"], 3),
                text=source["text"],
            )
            for source in sources
        ]

    @server.tool()
    async def ask_mondria(question: str) -> AskResult:
        """Answers a question about Mondria with the full RAG pipeline (search + LLM).

        Prefer search_mondria_docs when you can write the answer yourself; use this tool to get
        the same answer the public demo gives, in the language of the question, with the IDs of
        the sections it cites.

        Args:
            question: The question, in any language (max 300 characters).
        """
        result = await rag.answer_question(get_runtime(), question[:300])
        return AskResult(
            answer=result["answer"],
            cited_sections=[source["id"] for source in result["sources"] if source["cited"]],
        )

    return server


def run_stdio() -> None:
    """Local stdio server. Its runtime (OpenAI client, database) lives as long as the session."""
    configure_logging()
    holder: dict[str, Runtime] = {}

    @asynccontextmanager
    async def lifespan(_: MCPServer[Any]) -> AsyncIterator[None]:
        async with open_runtime() as runtime:
            holder["runtime"] = runtime
            yield

    build_mcp_server(lambda: holder["runtime"], lifespan=lifespan).run()
