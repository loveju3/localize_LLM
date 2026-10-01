import json
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class Repository:
    def __init__(self, settings):
        self.settings = settings
        self.workspace = settings.workspace_id

    def connect(self):
        self.settings.require("database_url")
        return psycopg.connect(self.settings.database_url.get_secret_value(),
                               row_factory=dict_row, connect_timeout=10)

    def initialize(self):
        spec = self.settings.embedding_spec()
        self.settings.require("r2_bucket_name", "r2_endpoint_url")
        with self.connect() as conn:
            conn.execute(Path(__file__).with_name("schema.sql").read_text())
            conn.execute("""INSERT INTO rag_workspaces
                (id, embedding_fingerprint, embedding_spec, bucket_name, storage_endpoint)
                VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                (self.workspace, self.settings.embedding_fingerprint(), Jsonb(spec),
                 self.settings.r2_bucket_name, self.settings.r2_endpoint_url.rstrip("/")))
            self._validate(conn)

    def _validate(self, conn):
        row = conn.execute("SELECT * FROM rag_workspaces WHERE id=%s", (self.workspace,)).fetchone()
        if not row:
            raise ValueError("工作區尚未初始化，請執行 python -m app.cli init-db")
        if row["embedding_fingerprint"] != self.settings.embedding_fingerprint():
            raise ValueError("本機 embedding 設定與雲端索引不一致；請同步設定，不能混用向量")
        if (row["bucket_name"] != self.settings.r2_bucket_name or
                row["storage_endpoint"] != self.settings.r2_endpoint_url.rstrip("/")):
            raise ValueError("本機 R2 設定與共用工作區不一致")
        return row

    def validate(self):
        with self.connect() as conn:
            return self._validate(conn)

    def documents(self):
        with self.connect() as conn:
            return conn.execute("""SELECT d.*, (SELECT count(*) FROM rag_chunks c
                WHERE c.document_id=d.id) AS chunk_count FROM rag_documents d
                WHERE workspace_id=%s ORDER BY created_at DESC""", (self.workspace,)).fetchall()

    def find_document(self, document_id=None, sha256=None):
        with self.connect() as conn:
            return conn.execute("""SELECT * FROM rag_documents WHERE workspace_id=%s
                AND (id=%s OR sha256=%s)""", (self.workspace, document_id, sha256)).fetchone()

    def save_document(self, document, chunks, vectors):
        with self.connect() as conn:
            self._validate(conn)
            row = conn.execute("""INSERT INTO rag_documents
                (id,workspace_id,sha256,filename,object_key,page_count,warnings)
                VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (workspace_id,sha256)
                DO NOTHING RETURNING *""", (document["id"], self.workspace,
                document["sha256"], document["filename"], document["object_key"],
                document["page_count"], Jsonb(document["warnings"]))).fetchone()
            if row is None:
                return conn.execute("SELECT * FROM rag_documents WHERE workspace_id=%s AND sha256=%s",
                                    (self.workspace, document["sha256"])).fetchone()
            if len(chunks) != len(vectors):
                raise ValueError("片段數與向量數不一致")
            with conn.cursor() as cur:
                cur.executemany("""INSERT INTO rag_chunks
                    (id,document_id,page_number,ordinal,content,embedding)
                    VALUES (%s,%s,%s,%s,%s,%s::vector)""", [
                    (str(uuid4()), document["id"], chunk["page"], i, chunk["text"],
                     json.dumps(vector)) for i, (chunk, vector) in enumerate(zip(chunks, vectors))])
            return row

    def search(self, vector, limit, document_ids):
        with self.connect() as conn:
            self._validate(conn)
            return conn.execute("""SELECT c.id, c.document_id, c.page_number, c.content,
                d.filename, 1-(c.embedding <=> %s::vector) AS similarity
                FROM rag_chunks c JOIN rag_documents d ON d.id=c.document_id
                WHERE d.workspace_id=%s AND (%s::uuid[] IS NULL OR d.id=ANY(%s::uuid[]))
                ORDER BY c.embedding <=> %s::vector, c.id LIMIT %s""",
                (json.dumps(vector), self.workspace, document_ids, document_ids,
                 json.dumps(vector), limit)).fetchall()

    def models(self):
        with self.connect() as conn:
            return conn.execute("""SELECT m.*, COALESCE(a.model_id=m.id,false) AS active
                FROM rag_models m LEFT JOIN rag_active_models a ON a.workspace_id=m.workspace_id
                WHERE m.workspace_id=%s ORDER BY m.created_at""", (self.workspace,)).fetchall()

    def model(self, model_id):
        with self.connect() as conn:
            return conn.execute("SELECT * FROM rag_models WHERE workspace_id=%s AND id=%s",
                                (self.workspace, model_id)).fetchone()

    def register_model(self, model):
        with self.connect() as conn:
            self._validate(conn)
            # Versions are immutable; do not silently overwrite an adapter in use.
            return conn.execute("""INSERT INTO rag_models
                (id,workspace_id,kind,base_model,base_revision,served_name,artifact_prefix,manifest)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                (model["id"], self.workspace, model["kind"], model["base_model"],
                 model["base_revision"], model["served_name"], model.get("artifact_prefix"),
                 Jsonb(model.get("manifest", {})))).fetchone()

    def activate(self, model_id):
        with self.connect() as conn:
            conn.execute("""INSERT INTO rag_active_models(workspace_id,model_id) VALUES (%s,%s)
                ON CONFLICT(workspace_id) DO UPDATE SET model_id=excluded.model_id,updated_at=now()""",
                (self.workspace, model_id))

    def active_model(self):
        with self.connect() as conn:
            return conn.execute("""SELECT m.* FROM rag_models m JOIN rag_active_models a
                ON a.workspace_id=m.workspace_id AND a.model_id=m.id
                WHERE m.workspace_id=%s""", (self.workspace,)).fetchone()

    def save_run(self, question, result, model, prompt_version):
        run_id = str(uuid4())
        with self.connect() as conn:
            conn.execute("""INSERT INTO rag_runs
                (id,workspace_id,question,result,model_snapshot,embedding_fingerprint,prompt_version)
                VALUES (%s,%s,%s,%s,%s,%s,%s)""", (run_id, self.workspace, question,
                Jsonb(result), Jsonb(model), self.settings.embedding_fingerprint(), prompt_version))
        return run_id
