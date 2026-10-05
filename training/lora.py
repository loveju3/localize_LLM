"""SFT with standard BF16 LoRA and durable JSONL/TensorBoard metrics."""
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    run_id: str
    model: str = "Qwen/Qwen3-8B"
    revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    rank: int = 8
    alpha: int = 16
    learning_rate: float = 1e-4
    epochs: float = 1
    max_steps: int = -1
    max_length: int = 1024
    gradient_accumulation: int = 4
    seed: int = 42

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.run_id):
            raise ValueError("Invalid run ID")
        if not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("A fixed model revision is required")
        if self.rank not in (4, 8, 16, 32) or self.alpha < 1:
            raise ValueError("Invalid LoRA rank/alpha")
        if not 0 < self.learning_rate <= 0.01 or not 0 < self.epochs <= 10:
            raise ValueError("Invalid learning rate/epochs")
        if self.max_steps != -1 and not 1 <= self.max_steps <= 10000:
            raise ValueError("Invalid max_steps")
        if not 128 <= self.max_length <= 2048 or not 1 <= self.gradient_accumulation <= 32:
            raise ValueError("Invalid sequence/batch settings")


def read_split(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows:
        raise ValueError("Empty split")
    for row in rows:
        messages = row.get("messages", [])
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError("Expected one system/user/assistant turn")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            raise ValueError("Empty message")
    return rows


def encode_sample(tokenizer, row, max_length):
    """Assert template prefix alignment; train only completion, including EOS."""
    messages = row["messages"]
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    if not full.startswith(prefix):
        raise ValueError("Chat template prefix mismatch; refusing an incorrect loss mask")
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoded["input_ids"]
    if len(tokens) > max_length:
        raise ValueError("Sample exceeds max_length; increase it or edit data, do not silently truncate answers")
    labels = []
    for token, (start, end) in zip(tokens, encoded["offset_mapping"]):
        if start < len(prefix) < end and full[start:len(prefix)].strip():
            raise ValueError("A token crosses a non-whitespace prompt boundary")
        labels.append(-100 if end <= len(prefix) else token)
    if not any(label != -100 for label in labels):
        raise ValueError("No supervised completion tokens")
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens), "labels": labels}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(config, data_dir, output_root, persist=lambda: None):
    config.validate()
    # Fail before allocating the model or accessing GPUs if data/run is invalid.
    output = Path(output_root) / config.run_id
    output.mkdir(parents=True, exist_ok=False)
    train_path, val_path = Path(data_dir) / "train.jsonl", Path(data_dir) / "validation.jsonl"
    rows, validation = read_split(train_path), read_split(val_path)
    metadata = {"config": asdict(config), "status": "starting", "dataset": {
        "train_sha256": file_hash(train_path), "validation_sha256": file_hash(val_path),
        "train_rows": len(rows), "validation_rows": len(validation)},
        "test_used_for_training": False, "method": "SFT + BF16 LoRA", "enable_thinking": False}
    meta_path = output / "run.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    persist()
    started = time.monotonic()
    try:
        import torch
        import transformers
        import peft
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA GPU required; CPU/Mac runtime is not supported")
        set_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train_data = Dataset.from_list([encode_sample(tokenizer, row, config.max_length) for row in rows])
        val_data = Dataset.from_list([encode_sample(tokenizer, row, config.max_length) for row in validation])
        model = AutoModelForCausalLM.from_pretrained(config.model, revision=config.revision,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=config.rank, lora_alpha=config.alpha,
            lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM", base_model_name_or_path=config.model, revision=config.revision))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if not trainable or any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
            raise RuntimeError("Unexpected trainable base parameters")
        metadata.update({"trainable_parameters": trainable, "total_parameters": total,
            "trainable_fraction": trainable / total, "gpu": torch.cuda.get_device_name(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}})
        torch.cuda.reset_peak_memory_stats()

        class Metrics(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                persist()

            def on_log(self, args, state, control, logs=None, **kwargs):
                values = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float)) and math.isfinite(v)}
                loss = values.get("eval_loss")
                if loss is not None and loss < 80:
                    values["eval_perplexity"] = math.exp(loss)
                values.update({"step": state.global_step, "epoch": state.epoch,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(values) + "\n")
                    stream.flush()
                meta_path.write_text(json.dumps({**metadata, "status": "running", "latest_metrics": values}, indent=2))
                persist()

        def collate(samples):
            size = max(len(s["input_ids"]) for s in samples)
            return {key: torch.tensor([s[key] + [pad] * (size - len(s[key])) for s in samples])
                    for key, pad in [("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)]}

        args = TrainingArguments(output_dir=str(output / "checkpoints"),
            num_train_epochs=config.epochs, max_steps=config.max_steps,
            learning_rate=config.learning_rate, per_device_train_batch_size=1,
            per_device_eval_batch_size=1, gradient_accumulation_steps=config.gradient_accumulation,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True, optim="adamw_torch", logging_steps=1, eval_strategy="steps", eval_steps=5,
            save_strategy="steps", save_steps=5, save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            report_to=["tensorboard"], logging_dir=str(output / "tensorboard"),
            seed=config.seed, data_seed=config.seed, remove_unused_columns=False)
        trainer = Trainer(model=model, args=args, train_dataset=train_data,
            eval_dataset=val_data, data_collator=collate, callbacks=[Metrics()])
        baseline_loss = trainer.evaluate()["eval_loss"]
        result = trainer.train()
        final = trainer.evaluate()
        adapter = output / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(output / "tokenizer")
        metadata.update({"status": "completed", "initial_eval_loss": baseline_loss,
            "final_eval_loss": final["eval_loss"], "train_metrics": result.metrics,
            "elapsed_seconds": time.monotonic() - started,
            "adapter_files": {p.name: file_hash(p) for p in adapter.iterdir() if p.is_file()},
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3})
        meta_path.write_text(json.dumps(metadata, indent=2))
        persist()
        return metadata
    except Exception as error:
        metadata.update({"status": "failed", "error_type": type(error).__name__,
                         "elapsed_seconds": time.monotonic() - started})
        meta_path.write_text(json.dumps(metadata, indent=2))
        persist()
        raise
