import secrets
from functools import lru_cache
from pathlib import Path
from uuid import UUID
from typing import Literal

import httpx
import psycopg
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import Settings
from app.db import Repository
from app.embeddings import Embedder
from app.inference import Inference
from app.service import RagService
from app.storage import ObjectStore
from app import lol_lab
from app import lora_ui
from app import training_ui

app = FastAPI(title="Localize LLM", version="0.1.0")
bearer = HTTPBearer(auto_error=False)


@lru_cache
def settings():
    return Settings()


@lru_cache
def service():
    config = settings()
    return RagService(config, Repository(config), ObjectStore(config), Embedder(config), Inference(config))


def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    expected = settings().app_api_key.get_secret_value()
    if not expected:
        raise HTTPException(503, "請先設定 APP_API_KEY")
    if not credentials or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(401, "API key 無效", headers={"WWW-Authenticate": "Bearer"})


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[UUID] | None = Field(default=None, max_length=50)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)

    @field_validator("content")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("訊息不可為空白")
        return value.strip()


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def valid_history(self):
        if sum(len(message.content) for message in self.messages) > 12000:
            raise ValueError("對話過長，請清除對話後重試")
        if self.messages[-1].role != "user" or any(
            message.role != ("user" if i % 2 == 0 else "assistant")
            for i, message in enumerate(self.messages)
        ):
            raise ValueError("對話必須從使用者開始、交替排列，並以使用者問題結尾")
        return self


@app.exception_handler(ValueError)
async def invalid_configuration(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(LookupError)
async def not_found(request, exc):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def upstream_failure(request, exc):
    # Do not return exception strings containing connection details, keys, or document text.
    return JSONResponse(status_code=502, content={
        "detail": "雲端服務連線或資料操作失敗，請檢查設定、資料庫初始化及服務狀態",
        "error_type": type(exc).__name__,
    })


for error_type in (psycopg.Error, httpx.HTTPError, BotoCoreError, ClientError):
    app.add_exception_handler(error_type, upstream_failure)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).with_name("lol.html"))


@app.get("/documents", include_in_schema=False)
def documents_index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/experiments/lol", include_in_schema=False)
def lol_index():
    return FileResponse(Path(__file__).with_name("lol.html"))


@app.get("/training", include_in_schema=False)
def training_index():
    return FileResponse(Path(__file__).with_name("training.html"))


class TrainingRequest(BaseModel):
    model_config = {"extra": "forbid"}
    rank: Literal[4, 8, 16, 32] = 8
    alpha: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.01)
    epochs: float = Field(default=1, gt=0, le=10)
    max_steps: int = Field(default=-1, ge=-1, le=10000)
    max_length: int = Field(default=1024, ge=128, le=2048)
    gradient_accumulation: int = Field(default=4, ge=1, le=32)
    seed: int = Field(default=42, ge=0, le=2147483647)


@app.post("/api/training", dependencies=[Depends(authorize)], status_code=202)
def start_training(body: TrainingRequest):
    return training_ui.start(body.model_dump())


@app.get("/api/training", dependencies=[Depends(authorize)])
def training_runs():
    return {"runs": training_ui.list_runs()}


@app.get("/api/training/{run_id}", dependencies=[Depends(authorize)])
def training_status(run_id: str):
    return training_ui.status(run_id)


@app.post("/api/training/{run_id}/publish", dependencies=[Depends(authorize)])
def publish_training(run_id: str):
    return training_ui.publish(service(), run_id)


@app.get("/api/training/{run_id}/download", dependencies=[Depends(authorize)])
def training_download(run_id: str, file: Literal["run.json", "metrics.jsonl", "adapter_config.json", "adapter_model.safetensors"]):
    relative = "adapter/" + file if file.startswith("adapter_") else file
    return Response(training_ui.read_bytes(run_id, relative), media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{file}"'})


@app.get("/api/lol/cases", dependencies=[Depends(authorize)])
def lol_cases(split: Literal["train", "validation", "test"] = "test",
              page: int = 1, page_size: int = 5):
    if page < 1 or not 1 <= page_size <= 20:
        raise HTTPException(400, "分頁範圍無效")
    rows = [row for row in lol_lab.cases() if row["split"] == split]
    return {"version": "lol-pilot-v1", "total": len(rows), "page": page,
            "cases": rows[(page - 1) * page_size:page * page_size]}


class LolRun(BaseModel):
    case_id: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=120)


@app.post("/api/lol/run", dependencies=[Depends(authorize)])
def lol_run(body: LolRun):
    return lol_lab.run_case(service(), body.case_id, body.model_id)


@app.get("/health")
def health():
    return {"status": "ok", "cloud_verified": False}


@app.get("/api/workspace", dependencies=[Depends(authorize)])
def workspace():
    row = service().repo.validate()
    return {"id": row["id"], "embedding_spec": row["embedding_spec"],
            "embedding_fingerprint": row["embedding_fingerprint"]}


@app.get("/api/documents", dependencies=[Depends(authorize)])
def documents():
    return service().repo.documents()


@app.post("/api/documents", dependencies=[Depends(authorize)])
def upload(file: UploadFile = File(...)):
    limit = settings().max_upload_mb * 1024 * 1024
    try:
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, "檔案超過大小限制")
        return service().ingest(file.filename or "document.pdf", data)
    finally:
        file.file.close()


@app.get("/api/documents/{document_id}/download", dependencies=[Depends(authorize)])
def download(document_id: UUID):
    service().repo.validate()
    document = service().repo.find_document(document_id=document_id)
    if not document:
        raise HTTPException(404, "文件不存在")
    return {"url": service().store.download_url(document["object_key"]), "expires_in": 300}


@app.post("/api/search", dependencies=[Depends(authorize)])
def search(body: Question):
    return {"sources": service().search(body.question.strip(), body.document_ids)}


@app.post("/api/ask", dependencies=[Depends(authorize)])
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(400, "問題不可為空白")
    return service().ask(body.question.strip(), body.document_ids)


@app.post("/api/chat", dependencies=[Depends(authorize)])
def chat(body: ChatRequest):
    return service().chat([message.model_dump() for message in body.messages])


@app.get("/api/models", dependencies=[Depends(authorize)])
def models():
    return service().model_statuses()


@app.post("/api/lora", dependencies=[Depends(authorize)])
def upload_adapter(model_id: str = Form(...), config_file: UploadFile = File(...),
                   weights_file: UploadFile = File(...)):
    try:
        return lora_ui.save_upload(service(), model_id, config_file, weights_file)
    finally:
        config_file.file.close()
        weights_file.file.close()


@app.post("/api/lora/{model_id}/load", dependencies=[Depends(authorize)])
def load_adapter(model_id: str):
    return lora_ui.load_adapter(service(), model_id)


@app.post("/api/models/{model_id}/activate", dependencies=[Depends(authorize)])
def activate(model_id: str):
    return service().activate(model_id)
