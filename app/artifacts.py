import hashlib
import json
import re
import shlex
from pathlib import Path
from uuid import uuid4


def validate_identity(model_id, base_model, base_revision):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", model_id):
        raise ValueError("模型 ID 僅接受英數字、底線與連字號，最多 80 字元")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", base_model):
        raise ValueError("基礎模型須使用 Hugging Face organization/model 名稱")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", base_revision):
        raise ValueError("基礎模型版本須為完整 40 位 commit SHA")


def upload_lora(repo, store, settings, directory, model_id, base_model, base_revision):
    validate_identity(model_id, base_model, base_revision)
    repo.validate()
    if repo.model(model_id):
        raise ValueError("此模型版本已登錄，請使用新的 ID")
    directory = Path(directory)
    config_bytes = (directory / "adapter_config.json").read_bytes()
    config = json.loads(config_bytes)
    if config.get("peft_type") != "LORA" or config.get("base_model_name_or_path") != base_model:
        raise ValueError("adapter_config 的 LoRA 類型或基礎模型不匹配")
    if config.get("revision") and config["revision"] != base_revision:
        raise ValueError("adapter_config 的基礎模型 revision 不匹配")
    if config.get("modules_to_save") or config.get("use_dora") or config.get("rank_pattern"):
        raise ValueError("第一版僅支援標準、固定 rank 的 LoRA；不支援額外儲存模組或 DoRA")
    rank = config.get("r")
    if type(rank) is not int or rank not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 512):
        raise ValueError("LoRA rank 不受此部署流程支援")
    weights_path = directory / "adapter_model.safetensors"
    if weights_path.stat().st_size > 2 * 1024 ** 3:
        raise ValueError("第一版 adapter 大小上限為 2GB")
    from safetensors import safe_open
    with safe_open(str(weights_path), framework="numpy") as weights:
        if not list(weights.keys()):
            raise ValueError("adapter 權重不可為空")
    files = {"adapter_config.json": config_bytes, "adapter_model.safetensors": weights_path.read_bytes()}
    prefix = f"workspaces/{settings.workspace_id}/adapters/{model_id}/{uuid4()}"
    manifest = {"base_model": base_model, "base_revision": base_revision,
                "rank": rank, "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    for name, data in files.items():
        store.put(f"{prefix}/{name}", data)
    store.put(f"{prefix}/manifest.json", json.dumps(manifest).encode(), "application/json")
    return repo.register_model({"id": model_id, "kind": "lora", "base_model": base_model,
        "base_revision": base_revision, "served_name": model_id, "artifact_prefix": prefix,
        "manifest": manifest})


def prepare_model(model, store, output):
    """Run on GPU host. Download only allowlisted files; verify registry checksums."""
    validate_identity(model["id"], model["base_model"], model["base_revision"])
    args = ["vllm", "serve", model["base_model"], "--revision", model["base_revision"],
            "--served-model-name", model["served_name"] if model["kind"] == "base" else model["base_model"],
            "--max-model-len", "8192", "--max-num-seqs", "1", "--host", "127.0.0.1"]
    if model["kind"] == "lora":
        manifest = model["manifest"]
        if (manifest.get("base_model") != model["base_model"] or
                manifest.get("base_revision") != model["base_revision"]):
            raise ValueError("LoRA manifest 與登錄版本不一致")
        expected = {"adapter_config.json", "adapter_model.safetensors"}
        if set(manifest["files"]) != expected:
            raise ValueError("LoRA manifest 檔案清單不受支援")
        contents = {}
        for name in sorted(expected):
            data = store.get(f"{model['artifact_prefix']}/{name}")
            if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
                raise ValueError("LoRA 檔案雜湊驗證失敗：" + name)
            contents[name] = data
        path = Path(output).resolve() / model["id"]
        path.mkdir(parents=True, exist_ok=False)
        for name, data in contents.items():
            (path / name).write_bytes(data)
        args += ["--enable-lora", "--max-lora-rank", str(manifest["rank"]), "--lora-modules",
                 json.dumps({"name": model["served_name"], "path": str(path),
                             "base_model_name": model["base_model"]})]
    return shlex.join(args)
