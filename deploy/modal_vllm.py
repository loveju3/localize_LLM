"""Authenticated OpenAI-compatible Qwen server; deploy with Modal CLI."""
import os
import subprocess
import json
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
IMAGE = "vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
app = modal.App("localize-llm-qwen3")
cache = modal.Volume.from_name("localize-llm-model-cache", create_if_missing=True)
adapters = modal.Volume.from_name("localize-llm-adapters", create_if_missing=True)
image = modal.Image.from_registry(IMAGE, add_python="3.12").entrypoint([]).pip_install("httpx==0.28.1", "fastapi==0.142.2").env({"HF_HOME": "/model-cache"})
download_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface-hub==0.36.0"
).env({"HF_HOME": "/model-cache"})


@app.function(image=download_image, volumes={"/model-cache": cache}, cpu=2, memory=4096,
              timeout=1800, max_containers=1)
def download_model():
    """Download once on CPU, validate configuration, persist before serving."""
    import json
    from pathlib import Path
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, revision=REVISION)
    json.loads((Path(path) / "config.json").read_text())
    cache.commit()
    return {"model": MODEL, "revision": REVISION}


@app.function(image=image, gpu=["L40S", "A100-40GB"], cpu=4, memory=16384,
              volumes={"/model-cache": cache, "/adapters": adapters},
              secrets=[modal.Secret.from_name("localize-llm-vllm")],
              min_containers=0, max_containers=1, scaledown_window=120,
              timeout=600, startup_timeout=900)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def serve():
    import asyncio
    import secrets
    import httpx
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, JSONResponse

    key = os.environ.get("VLLM_API_KEY")
    if not key:
        raise RuntimeError("VLLM_API_KEY secret is required")
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    args = ["vllm", "serve", MODEL, "--revision", REVISION,
            "--host", "127.0.0.1", "--port", "8000", "--dtype", "auto",
            "--enforce-eager", "--gpu-memory-utilization", "0.90",
            "--max-model-len", "8128", "--max-num-seqs", "1",
            "--enable-lora", "--max-lora-rank", "64", "--max-cpu-loras", "32"]
    saved = sorted(p for p in Path("/adapters").glob("*/adapter_config.json") if not p.parent.name.startswith("."))
    if saved:
        args += ["--lora-modules", *[json.dumps({"name": p.parent.name,
                  "path": str(p.parent), "base_model_name": MODEL}) for p in saved]]
    process = subprocess.Popen(args)

    @asynccontextmanager
    async def lifespan(api):
        async with httpx.AsyncClient(timeout=600) as client:
            for _ in range(420):
                if process.poll() is not None:
                    raise RuntimeError("vLLM startup failed")
                try:
                    r = await client.get("http://127.0.0.1:8000/health")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2)
            else:
                raise RuntimeError("vLLM startup timeout")
            api.state.client = client
            yield
        process.terminate()

    api = FastAPI(lifespan=lifespan)
    @api.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        if not secrets.compare_digest(request.headers.get("authorization", ""), "Bearer " + key):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        body = await request.body()
        if path == "v1/load_lora_adapter":
            payload = json.loads(body)
            name = payload.get("lora_name", "")
            import re
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", name) or payload.get("lora_path") != "/adapters/" + name:
                return JSONResponse({"detail": "Invalid adapter"}, status_code=400)
            await asyncio.to_thread(adapters.reload)
        elif path not in ("v1/models", "v1/chat/completions"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        response = await api.state.client.request(request.method,
            "http://127.0.0.1:8000/" + path, content=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        return Response(response.content, status_code=response.status_code,
                        media_type=response.headers.get("content-type", "application/json"))
    return api


@app.function(image=download_image.pip_install("httpx==0.28.1"),
              volumes={"/adapters": adapters}, cpu=1, memory=4096, timeout=600)
def stage_adapter(model_id, manifest, urls):
    import hashlib
    import re
    import tempfile
    import httpx
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", model_id):
        raise ValueError("Invalid adapter ID")
    expected = {"adapter_config.json", "adapter_model.safetensors"}
    if (manifest.get("base_model"), manifest.get("base_revision")) != (MODEL, REVISION) or set(urls) != expected or set(manifest["files"]) != expected or manifest["rank"] > 64:
        raise ValueError("Incompatible adapter")
    adapters.reload()
    target = Path("/adapters") / model_id
    if target.exists():
        for name in expected:
            if hashlib.sha256((target / name).read_bytes()).hexdigest() != manifest["files"][name]:
                raise ValueError("Adapter ID already has different contents")
        return
    with tempfile.TemporaryDirectory(dir="/adapters", prefix=".upload-") as temp:
        for name in expected:
            with httpx.Client(timeout=120) as client:
                response = client.get(urls[name])
                response.raise_for_status()
            data = response.content
            if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
                raise ValueError("Adapter checksum mismatch")
            (Path(temp) / name).write_bytes(data)
        Path(temp).rename(target)
    adapters.commit()


@app.local_entrypoint()
def main():
    print(download_model.remote())
