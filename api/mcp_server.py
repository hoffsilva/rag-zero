"""Local MCP server over stdio. Kept at this path because MCP clients are registered with it:
    claude mcp add mondria-rag -- <repo>/.venv/bin/python <repo>/api/mcp_server.py
The server itself lives in app/mcp.py (also served remotely at /api/mcp).
"""

from app.mcp import run_stdio

if __name__ == "__main__":
    run_stdio()
