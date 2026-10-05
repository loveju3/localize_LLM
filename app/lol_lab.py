"""Versioned LoL pilot cases; inference never receives reference answers."""
import json
import time
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "datasets/lol/v1"


def cases():
    return [json.loads(line) for line in (DATA / "catalog.jsonl").read_text().splitlines()]


def run_case(service, case_id, model_id):
    case = next((row for row in cases() if row["id"] == case_id), None)
    if not case:
        raise LookupError("題目不存在")
    service.repo.validate()
    model = service.repo.model(model_id)
    if not model:
        raise LookupError("模型不存在")
    if model["served_name"] not in service.inference.served_models():
        raise ValueError("模型已登錄但推論端點尚未載入；請先部署 adapter")
    start = time.monotonic()
    answer, usage = service.inference.chat(case["messages"][1:], model,
        system_prompt=case["messages"][0]["content"])
    groups = case["keyword_groups"]
    return {"case_id": case_id, "dataset_version": "lol-pilot-v1",
            "model": {k: model[k] for k in ("id", "base_model", "base_revision", "served_name")},
            "messages": case["messages"], "answer": answer, "usage": usage,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "keyword_match": all(any(term.casefold() in answer.casefold() for term in group)
                                 for group in groups) if groups else None,
            "reference_answer": case["reference_answer"], "source_url": case["source_url"],
            "manual_review": {"correct": None, "notes": ""}}
