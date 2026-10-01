import hashlib
import time
from pathlib import PurePosixPath
from uuid import uuid4

from app.documents import parse_pdf
from app.inference import PROMPT_VERSION, validate_answer


class RagService:
    def __init__(self, settings, repo, store, embedder, inference):
        self.settings, self.repo, self.store = settings, repo, store
        self.embedder, self.inference = embedder, inference

    def ingest(self, filename, data):
        self.repo.validate()
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise ValueError("檔案超過大小限制")
        digest = hashlib.sha256(data).hexdigest()
        existing = self.repo.find_document(sha256=digest)
        if existing:
            return {"document": existing, "deduplicated": True}
        chunks, pages, warnings = parse_pdf(data, self.settings)
        vectors = self.embedder.encode([chunk["text"] for chunk in chunks])
        key = f"workspaces/{self.settings.workspace_id}/documents/{digest}.pdf"
        self.store.put(key, data, "application/pdf")
        document = {"id": str(uuid4()), "sha256": digest,
            "filename": PurePosixPath(filename.replace("\\", "/")).name[:240] or "document.pdf",
            "object_key": key, "page_count": pages, "warnings": warnings}
        saved = self.repo.save_document(document, chunks, vectors)
        return {"document": saved, "deduplicated": str(saved["id"]) != document["id"]}

    def search(self, question, document_ids=None):
        self.repo.validate()
        vector = self.embedder.encode([question], query=True)[0]
        rows = self.repo.search(vector, self.settings.retrieval_k, document_ids)
        return [{"source_id": f"S{i + 1}", "chunk_id": str(row["id"]),
                 "document_id": str(row["document_id"]), "filename": row["filename"],
                 "page": row["page_number"], "text": row["content"],
                 "similarity": row["similarity"]}
                for i, row in enumerate(rows) if row["similarity"] >= self.settings.min_similarity]

    def model_statuses(self):
        models = self.repo.models()
        try:
            served = self.inference.served_models()
            error = None
        except Exception:
            served, error = set(), "無法連接 vLLM 或尚未設定；不能確認模型是否可用"
        for model in models:
            model["status"] = "unknown" if error else (
                "available" if model["served_name"] in served else "stored")
        return {"models": models, "inference_error": error}

    def activate(self, model_id):
        model = self.repo.model(model_id)
        if not model:
            raise LookupError("模型不存在")
        if model["served_name"] not in self.inference.served_models():
            raise ValueError("模型已登錄但 vLLM 尚未載入，請先部署／載入 adapter")
        self.repo.activate(model_id)
        return {"active_model_id": model_id}

    def ask(self, question, document_ids=None):
        started = time.monotonic()
        # Snapshot once: another user's selection cannot alter an in-flight request.
        model = self.repo.active_model()
        if not model:
            raise ValueError("尚未選擇可用的生成模型")
        snapshot = {key: model[key] for key in
                    ("id", "kind", "base_model", "base_revision", "served_name", "artifact_prefix")}
        sources = self.search(question, document_ids)
        usage = {}
        if sources:
            raw, usage = self.inference.answer(question, sources, model)
            result = validate_answer(raw, sources)
        else:
            result = validate_answer("{}", [])
        result.update({"model": snapshot, "usage": usage,
                       "elapsed_seconds": round(time.monotonic() - started, 3),
                       "prompt_version": PROMPT_VERSION})
        result["run_id"] = self.repo.save_run(question, result, snapshot, PROMPT_VERSION)
        return result
