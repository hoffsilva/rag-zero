-- Applied automatically by db.ensure_schema() before indexing; idempotent.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per document section (chunk) and embedding model. The key (section ID, model) lets
-- several models coexist, so indexing with one never wipes the vectors another instance or an
-- evaluation run is using, and re-running the indexing updates rows instead of duplicating them.
CREATE TABLE IF NOT EXISTS chunks (
    id              text NOT NULL,
    title           text NOT NULL,
    content         text NOT NULL,
    document_id     text NOT NULL,
    document_title  text NOT NULL,
    document_file   text NOT NULL,
    source_urls     text[] NOT NULL,
    embedding_model text NOT NULL,
    -- text-embedding-3-small returns 1536 dimensions; other text-embedding-3 models are
    -- asked for 1536 too (OpenAI "dimensions" parameter), so the column never changes.
    embedding       vector(1536) NOT NULL,
    PRIMARY KEY (id, embedding_model)
);

-- Migration for tables created when the primary key was the section ID alone.
DO $$
BEGIN
    IF (
        SELECT count(*)
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
        WHERE i.indrelid = 'chunks'::regclass AND i.indisprimary
    ) = 1 THEN
        ALTER TABLE chunks DROP CONSTRAINT chunks_pkey;
        ALTER TABLE chunks ADD PRIMARY KEY (id, embedding_model);
    END IF;
END
$$;

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
-- Pointless for 21 rows, but it is what keeps the search fast at millions of rows.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
