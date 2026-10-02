"""Opt-in live checks. Uploads a synthetic PDF and calls the configured cloud LLM."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.config import Settings


def check(client, pdf, record):
    def request(method, path, **kwargs):
        response = client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    record("health", request("GET", "/health"))
    record("workspace", request("GET", "/api/workspace"))
    models = request("GET", "/api/models")
    require(any(m["active"] and m["status"] == "available" for m in models["models"]), "active_model_unavailable")
    record("models", models)
    first = request("POST", "/api/chat", json={"messages": [
        {"role": "user", "content": "請記住這個測試代號：青松四十二。只回覆代號。"}]})
    require("青松四十二" in first["answer"], "chat_content_mismatch")
    record("chat_first_turn", first)
    second = request("POST", "/api/chat", json={"messages": [
        {"role": "user", "content": "請記住這個測試代號：青松四十二。只回覆代號。"},
        {"role": "assistant", "content": first["answer"]},
        {"role": "user", "content": "剛才的測試代號是什麼？只回覆代號。"}]})
    require("青松四十二" in second["answer"], "chat_history_mismatch")
    record("chat_second_turn", second)
    data = pdf.read_bytes()
    upload = request("POST", "/api/documents", files={"file": (pdf.name, data, "application/pdf")})
    document_id = upload["document"]["id"]
    record("upload", upload)
    duplicate = request("POST", "/api/documents", files={"file": (pdf.name, data, "application/pdf")})
    require(duplicate["deduplicated"] and duplicate["document"]["id"] == document_id, "deduplication_failed")
    record("deduplication", {"document_id": document_id, "deduplicated": True})
    listed = request("GET", "/api/documents")
    require(any(row["id"] == document_id for row in listed), "document_not_listed")
    record("document_list", {"test_document_present": True})
    download = request("GET", f"/api/documents/{document_id}/download")
    # Do not forward the application Authorization header to object storage.
    with httpx.Client(timeout=60) as external:
        response = external.get(download["url"])
        response.raise_for_status()
    require(hashlib.sha256(response.content).digest() == hashlib.sha256(data).digest(), "download_hash_mismatch")
    record("r2_download", {"sha256_matches": True})
    question = {"question": "Cedar Lantern 在 2025 年的營收成長率是多少？", "document_ids": [document_id]}
    found = request("POST", "/api/search", json=question)
    require(bool(found["sources"]), "retrieval_empty")
    require(all(s["document_id"] == document_id for s in found["sources"]), "document_scope_mismatch")
    record("search", found)
    answer = request("POST", "/api/ask", json=question)
    require(not answer["insufficient_evidence"] and bool(answer["citations"]), "grounded_answer_missing")
    require("20" in answer["answer"], "revenue_answer_mismatch")
    require(all(s["document_id"] == document_id and s["page"] == 1 for s in answer["citations"]), "citation_mismatch")
    record("grounded_answer", answer)
    refusal = request("POST", "/api/ask", json={"question": "Cedar Lantern 在 2035 年的確切營收是多少？",
                                                "document_ids": [document_id]})
    require(refusal["insufficient_evidence"] and not refusal["citations"], "expected_refusal_missing")
    record("refusal", refusal)


def main(argv=None):
    parser = argparse.ArgumentParser(description="真實雲端整合測試：上傳合成 PDF 並呼叫模型，可能產生費用")
    parser.add_argument("--pdf", required=True, type=Path, help="Cedar Lantern 合成測試 PDF，詳見 docs/cloud-validation.md")
    parser.add_argument("--output", required=True, type=Path, help="新的 JSONL 報告檔案")
    args = parser.parse_args(argv)
    if not args.pdf.is_file():
        parser.error("找不到合成測試 PDF")
    settings = Settings()
    settings.require("app_api_key")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.output.open("x", encoding="utf-8") as stream:
        def record(stage, data):
            stream.write(json.dumps({"stage": stage, "timestamp": datetime.now(timezone.utc).isoformat(),
                                     "data": data}, ensure_ascii=False) + "\n")
            stream.flush()
            print(stage, "OK", flush=True)

        try:
            with httpx.Client(base_url="http://127.0.0.1:8000", timeout=600,
                headers={"Authorization": "Bearer " + settings.app_api_key.get_secret_value()}) as client:
                check(client, args.pdf, record)
        except Exception as exc:
            # Never copy upstream exception bodies or signed object URLs into reports.
            detail = {"error_type": type(exc).__name__}
            if isinstance(exc, httpx.HTTPStatusError):
                detail["http_status"] = exc.response.status_code
            stream.write(json.dumps({"stage": "failure", "data": detail}) + "\n")
            print("FAIL", detail, flush=True)
            return 1
        record("complete", {"passed": True, "elapsed_seconds": round(time.monotonic() - started, 3)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
