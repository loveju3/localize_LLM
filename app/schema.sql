CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_workspaces (
    id text PRIMARY KEY,
    embedding_fingerprint text NOT NULL,
    embedding_spec jsonb NOT NULL,
    bucket_name text NOT NULL,
    storage_endpoint text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS rag_documents (
    id uuid PRIMARY KEY,
    workspace_id text NOT NULL REFERENCES rag_workspaces(id),
    sha256 text NOT NULL,
    filename text NOT NULL,
    object_key text NOT NULL,
    page_count integer NOT NULL,
    warnings jsonb NOT NULL DEFAULT '[]',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(workspace_id, sha256)
);
CREATE TABLE IF NOT EXISTS rag_chunks (
    id uuid PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    page_number integer NOT NULL,
    ordinal integer NOT NULL,
    content text NOT NULL,
    embedding vector NOT NULL,
    UNIQUE(document_id, ordinal)
);
CREATE INDEX IF NOT EXISTS rag_chunks_document ON rag_chunks(document_id);
CREATE TABLE IF NOT EXISTS rag_models (
    id text NOT NULL,
    workspace_id text NOT NULL REFERENCES rag_workspaces(id),
    kind text NOT NULL CHECK (kind IN ('base', 'lora')),
    base_model text NOT NULL,
    base_revision text NOT NULL,
    served_name text NOT NULL,
    artifact_prefix text,
    manifest jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(workspace_id, id),
    UNIQUE(workspace_id, served_name)
);
CREATE TABLE IF NOT EXISTS rag_active_models (
    workspace_id text PRIMARY KEY REFERENCES rag_workspaces(id),
    model_id text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY(workspace_id, model_id) REFERENCES rag_models(workspace_id, id)
);
CREATE TABLE IF NOT EXISTS rag_runs (
    id uuid PRIMARY KEY,
    workspace_id text NOT NULL REFERENCES rag_workspaces(id),
    question text NOT NULL,
    result jsonb NOT NULL,
    model_snapshot jsonb NOT NULL,
    embedding_fingerprint text NOT NULL,
    prompt_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
