"""Absolute paths to the files the API reads, independent of the module that reads them."""

from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
SCHEMA_FILE = API_DIR / "schema.sql"
DOCS_DIR = API_DIR / "mondria-rag-docs"
ENV_FILE = API_DIR / ".env.local"  # local development only; Vercel provides real env vars
