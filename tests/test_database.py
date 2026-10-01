"""Opt-in integration tests against a disposable PostgreSQL database with pgvector."""
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from app.config import Settings
from app.db import Repository


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "Set TEST_DATABASE_URL to a disposable pgvector database")
class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.config = Settings(_env_file=None, database_url=os.environ["TEST_DATABASE_URL"],
            workspace_id="test_" + uuid4().hex, embedding_revision="a" * 40,
            embedding_dimension=2, r2_bucket_name="test-bucket", r2_endpoint_url="https://example.invalid")
        self.repo = Repository(self.config)
        self.repo.initialize()

    def tearDown(self):
        with self.repo.connect() as conn:
            for table in ("rag_runs", "rag_active_models", "rag_models", "rag_documents"):
                conn.execute(f"DELETE FROM {table} WHERE workspace_id=%s", (self.repo.workspace,))
            conn.execute("DELETE FROM rag_workspaces WHERE id=%s", (self.repo.workspace,))

    def document(self, content_hash="a"):
        return {"id": str(uuid4()), "sha256": content_hash * 64, "filename": "report.pdf",
            "object_key": "documents/test.pdf", "page_count": 1, "warnings": []}

    def test_atomic_insert_search_filter_and_dedup(self):
        document = self.document()
        chunks = [{"page": 1, "text": "revenue grew 20%"}, {"page": 1, "text": "headcount"}]
        self.repo.save_document(document, chunks, [[1., 0.], [0., 1.]])
        rows = self.repo.search([1., 0.], 5, None)
        self.assertEqual(rows[0]["content"], "revenue grew 20%")
        self.assertAlmostEqual(rows[0]["similarity"], 1.)
        self.assertEqual(self.repo.search([1., 0.], 5, []), [])
        self.assertEqual(self.repo.search([1., 0.], 5, [uuid4()]), [])
        duplicate = self.repo.save_document(self.document(), chunks, [[1., 0.], [0., 1.]])
        self.assertEqual(str(duplicate["id"]), document["id"])
        self.assertEqual(self.repo.documents()[0]["chunk_count"], 2)

    def test_failed_chunks_roll_back_document(self):
        with self.assertRaises(ValueError):
            self.repo.save_document(self.document(), [{"page": 1, "text": "test"}], [])
        self.assertEqual(self.repo.documents(), [])

    def test_incompatible_embedding_settings_rejected(self):
        changed = self.config.model_copy(update={"embedding_revision": "b" * 40})
        with self.assertRaisesRegex(ValueError, "不一致"):
            Repository(changed).validate()
        with self.assertRaises(ValueError):
            Repository(changed).initialize()
        self.repo.validate()

    def test_other_workspace_cannot_retrieve_documents(self):
        document = self.document()
        self.repo.save_document(document, [{"page": 1, "text": "private report"}], [[1., 0.]])
        other_config = self.config.model_copy(update={"workspace_id": "other_" + uuid4().hex})
        other = Repository(other_config)
        other.initialize()
        try:
            self.assertEqual(other.documents(), [])
            self.assertEqual(other.search([1., 0.], 5, None), [])
            self.assertIsNone(other.find_document(document_id=document["id"]))
        finally:
            with other.connect() as conn:
                conn.execute("DELETE FROM rag_workspaces WHERE id=%s", (other.workspace,))

    def test_concurrent_duplicate_upload_commits_one_copy(self):
        def save(_):
            return self.repo.save_document(self.document(), [{"page": 1, "text": "report"}], [[1., 0.]])
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(save, range(2)))
        self.assertEqual(rows[0]["id"], rows[1]["id"])
        self.assertEqual(len(self.repo.documents()), 1)
        self.assertEqual(self.repo.documents()[0]["chunk_count"], 1)

    def test_model_selection_and_run_snapshot(self):
        model = {"id": "base-v1", "kind": "base", "base_model": "Qwen/Qwen3.5-4B",
                 "base_revision": "b" * 40, "served_name": "base-v1", "artifact_prefix": None}
        self.repo.register_model(model)
        self.repo.activate("base-v1")
        self.assertEqual(self.repo.active_model()["id"], "base-v1")
        self.assertTrue(self.repo.models()[0]["active"])
        run_id = self.repo.save_run("Question", {"answer": "Answer"}, model, "test-v1")
        with self.repo.connect() as conn:
            row = conn.execute("SELECT * FROM rag_runs WHERE id=%s", (run_id,)).fetchone()
        self.assertEqual(row["model_snapshot"]["base_revision"], "b" * 40)


if __name__ == "__main__":
    unittest.main()
