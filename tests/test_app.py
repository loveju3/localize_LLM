import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.artifacts import prepare_model, upload_lora
from app.config import Settings
from app.documents import parse_pdf
from app.inference import Inference, validate_answer
from app.main import app
from app.service import RagService


def config(**overrides):
    return Settings(_env_file=None, embedding_revision="a" * 40, **overrides)


def pdf_bytes(text="This report states that annual revenue grew by twenty percent."):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
        DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    data = BytesIO()
    writer.write(data)
    return data.getvalue()


class DocumentTests(unittest.TestCase):
    def test_pdf_extract_and_chunk(self):
        chunks, pages, warnings = parse_pdf(pdf_bytes("report " * 50), config(chunk_chars=100, chunk_overlap=20))
        self.assertEqual(pages, 1)
        self.assertFalse(warnings)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["page"] == 1 and len(c["text"]) <= 100 for c in chunks))

    def test_scanned_and_invalid_pdf_rejected(self):
        with self.assertRaises(ValueError):
            parse_pdf(b"not a pdf", config())
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        data = BytesIO()
        writer.write(data)
        with self.assertRaisesRegex(ValueError, "OCR"):
            parse_pdf(data.getvalue(), config())

    def test_fingerprint_changes_with_model_revision_and_prompt(self):
        self.assertNotEqual(config().embedding_fingerprint(),
            config(embedding_query_prompt="query: ").embedding_fingerprint())
        with self.assertRaises(ValueError):
            Settings(_env_file=None, embedding_revision="main").embedding_spec()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.repo, self.store, self.embed, self.llm = (MagicMock() for _ in range(4))
        self.rag = RagService(config(), self.repo, self.store, self.embed, self.llm)
        self.model = {"id": "base-v1", "kind": "base", "base_model": "Qwen/Qwen3.5-4B",
                      "base_revision": "b" * 40, "served_name": "base-v1", "artifact_prefix": None}
        self.repo.active_model.return_value = self.model
        self.repo.save_run.return_value = "run-1"
        self.repo.search.return_value = [{"id": "chunk-1", "document_id": "doc-1",
            "page_number": 2, "content": "Revenue grew 20%.", "filename": "report.pdf", "similarity": .8}]

    def test_dedup_does_not_upload_or_embed(self):
        self.repo.find_document.return_value = {"id": "already-there"}
        result = self.rag.ingest("report.pdf", pdf_bytes())
        self.assertTrue(result["deduplicated"])
        self.embed.encode.assert_not_called()
        self.store.put.assert_not_called()

    def test_ingestion_persists_pages_vectors_and_original(self):
        self.repo.find_document.return_value = None
        self.embed.encode.return_value = [[1., 0.]]
        self.repo.save_document.side_effect = lambda doc, chunks, vectors: doc
        result = self.rag.ingest("../report.pdf", pdf_bytes())
        self.assertEqual(result["document"]["filename"], "report.pdf")
        self.store.put.assert_called_once()
        self.assertEqual(self.repo.save_document.call_args.args[1][0]["page"], 1)

    def test_untrusted_citation_is_refused(self):
        self.llm.answer.return_value = (json.dumps({"answer": "Invented", "source_ids": ["S999"],
                                                  "insufficient_evidence": False}), {})
        result = self.rag.ask("Revenue?")
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["citations"], [])

    def test_answer_uses_original_text_and_model_snapshot(self):
        self.llm.answer.return_value = (json.dumps({"answer": "成長 20%", "source_ids": ["S1"],
                                                  "insufficient_evidence": False}), {"total_tokens": 42})
        result = self.rag.ask("Revenue?")
        self.assertEqual(result["citations"][0]["page"], 2)
        self.assertEqual(result["model"]["id"], "base-v1")
        self.assertEqual(self.llm.answer.call_args.args[1][0]["text"], "Revenue grew 20%.")
        self.repo.active_model.assert_called_once()

    def test_no_evidence_skips_paid_inference(self):
        self.repo.search.return_value = []
        self.assertTrue(self.rag.ask("Unknown?")["insufficient_evidence"])
        self.llm.answer.assert_not_called()

    def test_unloaded_model_cannot_be_activated(self):
        self.repo.model.return_value = self.model
        self.llm.served_models.return_value = set()
        with self.assertRaises(ValueError):
            self.rag.activate("base-v1")
        self.repo.activate.assert_not_called()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config(app_api_key="test-key")
        self.fake = MagicMock()
        app.dependency_overrides.clear()
        self.settings_patch = patch("app.main.settings", return_value=self.cfg)
        self.service_patch = patch("app.main.service", return_value=self.fake)
        self.settings_patch.start()
        self.service_patch.start()
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer test-key"}

    def tearDown(self):
        self.settings_patch.stop()
        self.service_patch.stop()

    def test_health_and_ui_without_credentials(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertIn("文件問答", self.client.get("/").text)

    def test_protected_routes_require_key(self):
        self.assertEqual(self.client.get("/api/documents").status_code, 401)
        self.fake.repo.documents.return_value = []
        self.assertEqual(self.client.get("/api/documents", headers=self.headers).json(), [])

    def test_empty_question_rejected(self):
        response = self.client.post("/api/ask", headers=self.headers, json={"question": "  "})
        self.assertEqual(response.status_code, 400)

    def test_upload_size_limit(self):
        self.cfg.max_upload_mb = 1
        result = self.client.post("/api/documents", headers=self.headers,
            files={"file": ("large.pdf", b"x" * (1024 * 1024 + 1), "application/pdf")})
        self.assertEqual(result.status_code, 413)
        self.fake.ingest.assert_not_called()

    def test_upstream_errors_do_not_leak_secrets(self):
        self.fake.model_statuses.side_effect = httpx.ConnectError("secret-password")
        response = self.client.get("/api/models", headers=self.headers)
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret-password", response.text)


class ArtifactTests(unittest.TestCase):
    def test_corrupted_download_does_not_write(self):
        model = {"id": "adapter-v1", "kind": "lora", "base_model": "Qwen/Qwen3-8B",
            "base_revision": "a" * 40, "served_name": "adapter-v1", "artifact_prefix": "prefix",
            "manifest": {"base_model": "Qwen/Qwen3-8B", "base_revision": "a" * 40, "rank": 8,
                "files": {name: "0" * 64 for name in ("adapter_config.json", "adapter_model.safetensors")}}}
        store = MagicMock()
        store.get.return_value = b"corrupt"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "雜湊"):
                prepare_model(model, store, directory)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_base_model_mismatch_prevents_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "base_model_name_or_path": "Other/Model"}))
            repo, store = MagicMock(), MagicMock()
            repo.model.return_value = None
            with self.assertRaisesRegex(ValueError, "不匹配"):
                upload_lora(repo, store, config(), directory, "adapter-v1", "Qwen/Qwen3-8B", "a" * 40)
            store.put.assert_not_called()

    def test_invalid_llm_json_refused(self):
        for raw in ("not json", "[]", '{"answer":"x","source_ids":[{}]}'):
            self.assertTrue(validate_answer(raw, [])["insufficient_evidence"])

    def test_lora_upload_download_round_trip(self):
        import numpy as np
        from safetensors.numpy import save_file
        objects = {}
        store, repo = MagicMock(), MagicMock()
        store.put.side_effect = lambda key, data, *args: objects.update({key: data})
        store.get.side_effect = objects.__getitem__
        repo.model.return_value = None
        repo.register_model.side_effect = lambda model: model
        with tempfile.TemporaryDirectory() as directory:
            training = Path(directory, "training")
            training.mkdir()
            Path(training, "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen3-8B", "r": 8}))
            save_file({"test.lora_A.weight": np.ones((8, 2), dtype=np.float32)},
                      str(training / "adapter_model.safetensors"))
            model = upload_lora(repo, store, config(), training, "report-v1", "Qwen/Qwen3-8B", "a" * 40)
            command = prepare_model(model, store, Path(directory, "download"))
            self.assertIn("--enable-lora", command)
            self.assertIn("--max-lora-rank 8", command)
            self.assertEqual(Path(directory, "download/report-v1/adapter_model.safetensors").read_bytes(),
                             (training / "adapter_model.safetensors").read_bytes())


class InferenceTests(unittest.TestCase):
    def test_http_payload_contains_text_and_selected_model(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "adapter-v1"}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}],
                                            "usage": {"total_tokens": 15}})
        client = Inference(config(vllm_base_url="https://example.invalid/v1", vllm_api_key="test"))
        def connection():
            return httpx.Client(base_url="https://example.invalid/v1/", transport=httpx.MockTransport(handler))
        with patch.object(client, "client", side_effect=connection):
            self.assertEqual(client.served_models(), {"adapter-v1"})
            client.answer("營收？", [{"source_id": "S1", "text": "營收成長20%"}],
                          {"served_name": "adapter-v1", "base_model": "Qwen/Qwen3-8B"})
        payload = json.loads(requests[1].content)
        self.assertEqual(requests[1].url.path, "/v1/chat/completions")
        self.assertEqual(payload["model"], "adapter-v1")
        self.assertIn("營收成長20%", payload["messages"][1]["content"])
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
