"""Batch RAG evaluation with frozen retrieval per question and local JSONL reports."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from app.inference import PROMPT_VERSION, SYSTEM_PROMPT, validate_answer


class ExpectedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: UUID
    page: int = Field(strict=True, ge=1)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[UUID] | None = Field(default=None, min_length=1, max_length=50)
    expected_refusal: StrictBool
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    answer_contains: list[str] = Field(default_factory=list)
    reference_answer: str = ""

    @field_validator("answer_contains")
    @classmethod
    def nonempty_terms(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("answer_contains 不可包含空白字串")
        return [value.strip() for value in values]

    @model_validator(mode="after")
    def coherent_expectations(self):
        if self.expected_refusal and (self.expected_sources or self.answer_contains):
            raise ValueError("拒答題不可指定預期引用或答案關鍵字")
        if self.document_ids and any(s.document_id not in self.document_ids for s in self.expected_sources):
            raise ValueError("預期引用必須位於 document_ids 範圍內")
        return self


def load_cases(path):
    data = Path(path).read_bytes()
    cases = []
    for number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate_json(line))
        except ValueError:
            raise ValueError(f"題庫第 {number} 行格式或欄位無效") from None
    if not cases:
        raise ValueError("題庫不可為空")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("題目 id 不可重複")
    return cases, hashlib.sha256(data).hexdigest()


def score(case, sources, result):
    expected = {(str(s.document_id), s.page) for s in case.expected_sources}

    def recall(rows):
        if not expected:
            return None
        actual = {(str(s["document_id"]), s["page"]) for s in rows}
        return len(actual & expected) / len(expected)

    return {
        "refusal_correct": result["insufficient_evidence"] == case.expected_refusal,
        "retrieval_page_recall": recall(sources),
        "citation_page_recall": recall(result["citations"]),
        "answer_terms_present": (all(term.casefold() in result["answer"].casefold()
                                     for term in case.answer_contains)
                                 if case.answer_contains else None),
    }


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault(record["model_id"], []).append(record)
    summary = {}
    for model_id, rows in groups.items():
        successful = [row for row in rows if row["status"] == "ok"]
        metrics = {}
        for name in ("refusal_correct", "retrieval_page_recall", "citation_page_recall", "answer_terms_present"):
            values = [row["metrics"][name] for row in successful if row["metrics"][name] is not None]
            metrics[name] = {"mean": sum(values) / len(values) if values else None, "count": len(values)}
        summary[model_id] = {"total": len(rows), "completed": len(successful),
                             "errors": len(rows) - len(successful), "metrics": metrics}
    return summary


def run_evaluation(cases, rag, model_ids, emit):
    """Retrieve once per case; never change the workspace's active model."""
    rag.repo.validate()
    models = []
    served = rag.inference.served_models()
    for model_id in model_ids:
        model = rag.repo.model(model_id)
        if not model or model["served_name"] not in served:
            raise ValueError("指定模型不存在或尚未由 vLLM 載入")
        models.append({key: model[key] for key in
                       ("id", "kind", "base_model", "base_revision", "served_name", "artifact_prefix")})
    emit({"type": "configuration", "workspace_id": rag.settings.workspace_id,
          "embedding_fingerprint": rag.settings.embedding_fingerprint(),
          "retrieval_k": rag.settings.retrieval_k, "min_similarity": rag.settings.min_similarity,
          "prompt_version": PROMPT_VERSION,
          "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), "models": models})
    records = []
    for case in cases:
        started = time.monotonic()
        try:
            sources = rag.search(case.question, case.document_ids)
            retrieval_error = None
        except Exception as exc:
            sources, retrieval_error = [], type(exc).__name__
        retrieval_seconds = time.monotonic() - started
        for model in models:
            record = {"type": "result", "case_id": case.id, "category": case.category,
                      "case": case.model_dump(mode="json"), "model_id": model["id"],
                      "model": model, "retrieved_sources": sources,
                      "retrieval_seconds": round(retrieval_seconds, 3)}
            started = time.monotonic()
            if retrieval_error:
                record.update(status="error", error_stage="retrieval", error_type=retrieval_error)
            else:
                try:
                    raw, usage = rag.inference.answer(case.question, sources, model) if sources else ("{}", {})
                    result = validate_answer(raw, sources)
                    record.update(status="ok", result=result, raw_answer=raw, usage=usage,
                                  metrics=score(case, sources, result),
                                  manual_review={"answer_correct": None, "citations_supported": None,
                                                 "notes": ""})
                except Exception as exc:
                    # Exception messages may contain API credentials or connection strings.
                    record.update(status="error", error_stage="inference", error_type=type(exc).__name__)
            record["inference_seconds"] = round(time.monotonic() - started, 3)
            records.append(record)
            emit(record)
    summary = summarize(records)
    emit({"type": "summary", "models": summary})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="文件問答批次評估；執行時會呼叫已啟動的雲端模型")
    parser.add_argument("dataset", help="JSONL 題庫")
    parser.add_argument("--validate-only", action="store_true", help="只驗證題庫，不連線或下載模型")
    parser.add_argument("--model", action="append", help="已登錄及載入的模型 ID，可重複指定")
    parser.add_argument("--output", help="新的 JSONL 報告路徑，建議 data/evaluations/run.jsonl")
    args = parser.parse_args(argv)
    try:
        cases, digest = load_cases(args.dataset)
    except (ValueError, OSError):
        parser.error("無法讀取題庫，或題庫格式無效；請檢查 JSONL 欄位與唯一 id")
    if args.validate_only:
        print(f"題庫有效：{len(cases)} 題；SHA-256：{digest}")
        return 0
    if not args.model or not args.output:
        parser.error("執行評估需要 --model 與 --output")
    if len(set(args.model)) != len(args.model):
        parser.error("--model 不可重複")
    # Exclusive creation prevents overwrites, and checks the output before cloud work.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        def emit(record):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

        emit({"type": "header", "schema_version": 1, "dataset_sha256": digest,
              "case_count": len(cases), "created_at": datetime.now(timezone.utc).isoformat()})
        try:
            from app.main import service
            summary = run_evaluation(cases, service(), args.model, emit)
        except Exception as exc:
            emit({"type": "fatal_error", "error_type": type(exc).__name__})
            print("評估未完成，請檢查模型及工作區設定；錯誤類型已寫入報告。")
            return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if any(row["errors"] for row in summary.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
