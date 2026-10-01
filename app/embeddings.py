import os
from threading import Lock

import numpy as np


class Embedder:
    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._lock = Lock()

    def encode(self, texts, query=False):
        with self._lock:
            spec = self.settings.embedding_spec()
            if self._model is None:
                os.environ.setdefault("HF_HOME", self.settings.hf_home)
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(spec["model"], revision=spec["revision"],
                    device=self.settings.embedding_device, trust_remote_code=False)
                self._model.max_seq_length = spec["max_tokens"]
            prompt = spec["query_prompt"] if query else spec["document_prompt"]
            # Reject silent truncation, which would leave stored text outside the embedding.
            for text in texts:
                if len(self._model.tokenizer.encode(prompt + text)) > spec["max_tokens"]:
                    raise ValueError("文字超過 embedding token 限制，請縮短問題／CHUNK_CHARS")
            vectors = self._model.encode(texts, prompt=prompt, batch_size=8,
                normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(texts), spec["dimension"]):
                raise ValueError("embedding 輸出維度與索引設定不符")
            if not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
                raise ValueError("embedding 含有無效數值")
            return vectors.tolist()
