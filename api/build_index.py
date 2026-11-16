"""Indexes mondria-rag-docs/ into Postgres (table chunks) with the configured embedding model.

Idempotent: running it twice updates the same rows instead of duplicating them.
The API also indexes on its first request when the table has no chunks for the model.
Run from api/ (reads .env.local):  python build_index.py
"""

from app.cli import index

if __name__ == "__main__":
    index()
