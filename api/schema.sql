-- Applied automatically by db.ensure_schema() before indexing; idempotent.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per document section (chunk), collection and embedding model.
-- - collection: "mondria" for the demo knowledge base, "upload-<random hex>" for a visitor's
--   uploaded document, which only that visitor can query and which expires after 24 h.
-- - embedding_model: several models coexist, so indexing with one never wipes the vectors
--   another instance or an evaluation run is using.
-- Re-running the indexing updates rows instead of duplicating them.
CREATE TABLE IF NOT EXISTS chunks (
    collection      text NOT NULL DEFAULT 'mondria',
    created_at      timestamptz NOT NULL DEFAULT now(),
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
    PRIMARY KEY (collection, id, embedding_model)
);

-- Migrations for tables created before collections (primary key on fewer columns).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS collection text NOT NULL DEFAULT 'mondria';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
DO $$
BEGIN
    IF (
        SELECT count(*)
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
        WHERE i.indrelid = 'chunks'::regclass AND i.indisprimary
    ) < 3 THEN
        ALTER TABLE chunks DROP CONSTRAINT chunks_pkey;
        ALTER TABLE chunks ADD PRIMARY KEY (collection, id, embedding_model);
    END IF;
END
$$;

-- One row per uploaded document: what the /documents page shows next to the questions.
CREATE TABLE IF NOT EXISTS uploads (
    collection          text PRIMARY KEY,
    title               text NOT NULL,
    file                text NOT NULL,
    kind                text NOT NULL,
    size_bytes          integer NOT NULL,
    sections            text[] NOT NULL,
    language            text NOT NULL,
    summary             text NOT NULL,
    suggested_questions text[] NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now()
);

-- Full-text search for the keyword and hybrid strategies: a French tsvector kept up to date by
-- Postgres itself (generated column), with a GIN index.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('french', title || ' ' || content)) STORED;
CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin ON chunks USING gin (content_tsv);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
-- Pointless for 21 rows, but it is what keeps the search fast at millions of rows.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
