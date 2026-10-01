import argparse
import json

from app.artifacts import prepare_model, upload_lora, validate_identity
from app.config import Settings
from app.db import Repository
from app.storage import ObjectStore


def main():
    parser = argparse.ArgumentParser(description="共用 RAG 工作區管理")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="建立資料表及不可混用的 embedding 設定")
    resolve = commands.add_parser("resolve-revision", help="查詢 Hugging Face 模型 commit，不下載權重")
    resolve.add_argument("model")
    for name in ("register-base", "upload-lora"):
        command = commands.add_parser(name)
        command.add_argument("--id", required=True)
        command.add_argument("--base-model", required=True)
        command.add_argument("--base-revision", required=True)
        if name == "upload-lora":
            command.add_argument("--directory", required=True)
    prepare = commands.add_parser("prepare-model", help="在 GPU 主機下載 adapter 並輸出 vLLM 啟動指令")
    prepare.add_argument("--id", required=True)
    prepare.add_argument("--output", default="models")
    args = parser.parse_args()
    config = Settings()
    repo, store = Repository(config), ObjectStore(config)
    if args.command == "resolve-revision":
        from huggingface_hub import model_info
        print(model_info(args.model).sha)
    elif args.command == "init-db":
        repo.initialize()
        print("工作區已初始化：" + config.workspace_id)
    elif args.command == "register-base":
        validate_identity(args.id, args.base_model, args.base_revision)
        result = repo.register_model({"id": args.id, "kind": "base", "base_model": args.base_model,
            "base_revision": args.base_revision, "served_name": args.id})
        print(json.dumps(result, default=str, ensure_ascii=False))
    elif args.command == "upload-lora":
        result = upload_lora(repo, store, config, args.directory, args.id, args.base_model, args.base_revision)
        print(json.dumps(result, default=str, ensure_ascii=False))
    elif args.command == "prepare-model":
        model = repo.model(args.id)
        if not model:
            raise ValueError("模型不存在")
        print(prepare_model(model, store, args.output))
        print("請先設定 VLLM_API_KEY；此指令未啟動 GPU 或 vLLM。遠端存取請配置 HTTPS proxy 或 SSH tunnel。")


if __name__ == "__main__":
    main()
