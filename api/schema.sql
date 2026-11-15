-- Applied automatically by db.ensure_schema() before indexing; idempotent.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per document section (chunk). The section ID ("MON-01-04") is the primary key,
-- so re-running the indexing updates rows instead of duplicating them.
CREATE TABLE IF NOT EXISTS chunks (
    id              text PRIMARY KEY,
    title           text NOT NULL,
    content         text NOT NULL,
    document_id     text NOT NULL,
    document_title  text NOT NULL,
    document_file   text NOT NULL,
    source_urls     text[] NOT NULL,
    embedding_model text NOT NULL,
    -- text-embedding-3-small returns 1536 dimensions; other text-embedding-3 models are
    -- asked for 1536 too (OpenAI "dimensions" parameter), so the column never changes.
    embedding       vector(1536) NOT NULL
);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
-- Pointless for 21 rows, but it is what keeps the search fast at millions of rows.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
