import hashlib
import json
import re

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: SecretStr = SecretStr("")
    app_api_key: SecretStr = SecretStr("")
    workspace_id: str = Field(default="personal", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    r2_endpoint_url: str = ""
    r2_access_key_id: SecretStr = SecretStr("")
    r2_secret_access_key: SecretStr = SecretStr("")
    r2_bucket_name: str = ""
    vllm_base_url: str = ""
    vllm_api_key: SecretStr = SecretStr("")
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = ""
    embedding_dimension: int = Field(default=1024, ge=1, le=4096)
    embedding_device: str = "cpu"
    embedding_max_tokens: int = Field(default=2048, ge=128, le=8192)
    embedding_query_prompt: str = ""
    embedding_document_prompt: str = ""
    hf_home: str = ".cache/huggingface"
    max_upload_mb: int = Field(default=20, ge=1, le=100)
    max_pdf_pages: int = Field(default=300, ge=1)
    max_chunks: int = Field(default=3000, ge=1)
    chunk_chars: int = Field(default=1000, ge=100, le=2000)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_k: int = Field(default=5, ge=1, le=8)
    min_similarity: float = Field(default=0.35, ge=-1, le=1)
    llm_timeout_seconds: int = Field(default=180, ge=1)

    def require(self, *names):
        missing = []
        for name in names:
            value = getattr(self, name)
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
            if not value:
                missing.append(name.upper())
        if missing:
            raise ValueError("尚未設定：" + ", ".join(missing))

    def embedding_spec(self):
        if not re.fullmatch(r"[a-fA-F0-9]{40}", self.embedding_revision):
            raise ValueError("EMBEDDING_REVISION 必須是模型的完整 40 位 commit SHA")
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("CHUNK_OVERLAP 必須小於 CHUNK_CHARS")
        return {
            "model": self.embedding_model, "revision": self.embedding_revision.lower(),
            "dimension": self.embedding_dimension, "normalize": True,
            "max_tokens": self.embedding_max_tokens,
            "query_prompt": self.embedding_query_prompt,
            "document_prompt": self.embedding_document_prompt,
            "encoding": "sentence-transformers-encode-v1",
        }

    def embedding_fingerprint(self):
        value = json.dumps(self.embedding_spec(), sort_keys=True).encode()
        return hashlib.sha256(value).hexdigest()
