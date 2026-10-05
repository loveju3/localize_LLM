"""Separate on-demand LoRA job; does not alter the live inference deployment."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("localize-llm-lora-training")
cache = modal.Volume.from_name("localize-llm-model-cache", create_if_missing=True)
runs = modal.Volume.from_name("localize-llm-training-runs", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements-training.txt"))
    .env({"HF_HOME": "/model-cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(ROOT / "training"), remote_path="/root/training")
    .add_local_file(str(ROOT / "datasets/lol/v1/train.jsonl"), remote_path="/dataset/train.jsonl")
    .add_local_file(str(ROOT / "datasets/lol/v1/validation.jsonl"), remote_path="/dataset/validation.jsonl"))


@app.function(image=image, gpu=["L40S", "A100-40GB"], cpu=4, memory=32768,
              volumes={"/model-cache": cache, "/runs": runs}, timeout=3600,
              max_containers=1, retries=0)
def train_job(config):
    from training.lora import TrainConfig, train
    return train(TrainConfig(**config), "/dataset", "/runs", persist=runs.commit)


@app.local_entrypoint()
def main(run_id: str, smoke: bool = False, rank: int = 8, epochs: float = 1):
    from dataclasses import asdict
    from training.lora import TrainConfig
    config = TrainConfig(run_id=run_id, rank=rank, alpha=rank * 2, epochs=epochs,
                         max_steps=2 if smoke else -1)
    config.validate()
    print(train_job.remote(asdict(config)))
