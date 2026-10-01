import secrets
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import httpx
import psycopg
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.config import Settings
from app.db import Repository
from app.embeddings import Embedder
from app.inference import Inference
from app.service import RagService
from app.storage import ObjectStore

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
    return FileResponse(Path(__file__).with_name("index.html"))


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


@app.get("/api/models", dependencies=[Depends(authorize)])
def models():
    return service().model_statuses()


@app.post("/api/models/{model_id}/activate", dependencies=[Depends(authorize)])
def activate(model_id: str):
    return service().activate(model_id)
