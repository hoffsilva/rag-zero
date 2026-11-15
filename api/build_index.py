"""Builds data/index.json from mondria-rag-docs/ with the configured embedding model.

Committing the index avoids re-embedding the docs on every cold start. If it is missing or
was built with another model, the API rebuilds it in memory instead (see rag.get_index).

Run from this folder with the key in the environment:
    set -a && source .env.local && set +a && python build_index.py
"""

import asyncio
import json

from openai_client import embedding_model
from rag import INDEX_FILE, build_index


async def main() -> None:
    index = await build_index(embedding_model())
    INDEX_FILE.parent.mkdir(exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index) + "\n", encoding="utf-8")
    print(f"Indexed {len(index['chunks'])} chunks with {index['model']} into data/index.json")


if __name__ == "__main__":
    asyncio.run(main())
