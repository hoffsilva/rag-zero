"""MCP server exposing the Mondria knowledge base to MCP clients such as Claude Code.

Runs over stdio: the client starts this script as a subprocess and exchanges JSON-RPC
messages through its stdin/stdout. Nothing else may be printed to stdout.

Register it in Claude Code:
    claude mcp add mondria-rag -- <repo>/.venv/bin/python <repo>/api/mcp_server.py
"""

import os
from pathlib import Path
from typing import TypedDict

from mcp.server.mcpserver import MCPServer

import rag


def load_env_file(path: Path) -> None:
    """Loads KEY=VALUE lines from .env.local so the client doesn't have to pass secrets.
    Variables already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_env_file(Path(__file__).parent / ".env.local")

server = MCPServer(
    name="mondria-rag",
    instructions=(
        "Knowledge base about Mondria, a Québec company offering managed IT, cybersecurity and AI "
        "services to SMEs (offices, services, Microsoft 365, CIS-based security, Etudia for notaries, "
        "sectors, FAQ). The documents are in French. Cite section IDs such as [MON-01-04]."
    ),
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


@server.tool()
async def search_mondria_docs(query: str, top_k: int = 4) -> list[SearchResult]:
    """Semantic search over the Mondria knowledge base (French documents).

    Use it for any question about Mondria: offices and contact details, managed IT services,
    Microsoft 365, cybersecurity, AI offers, Etudia for notaries, sectors and FAQ.
    Returns the closest sections with their ID, document title, cosine similarity score
    (higher is closer) and full text. Write the answer yourself from the returned text and
    cite the section IDs. A low score on every result means the answer is probably not in
    the knowledge base.

    Args:
        query: The question or keywords, in any language.
        top_k: How many sections to return (1 to 10).
    """
    sources = await rag.retrieve(query, top_k=max(1, min(top_k, 10)))
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
    result = await rag.answer_question(question[:300])
    return AskResult(
        answer=result["answer"],
        cited_sections=[source["id"] for source in result["sources"] if source["cited"]],
    )


if __name__ == "__main__":
    server.run()
