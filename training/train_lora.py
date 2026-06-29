#!/usr/bin/env python3
"""QLoRA / LoRA fine-tune Qwen2-0.5B-Instruct on LLM4OSC intent JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_TRAIN = "data/train.jsonl"
DEFAULT_VAL = "data/val.jsonl"
DEFAULT_OUTPUT = "models/qwen2-0.5b-osc"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _format_row(tokenizer, messages: list[dict[str, str]]) -> dict[str, str]:
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def _checkpoint_ok(
    ckpt_dir: Path,
    base_model: str,
    probe_messages: list[dict[str, str]],
) -> bool:
    """Return True if adapter checkpoint emits parseable JSON (not collapsed)."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from llm4osc.llm import extract_json_object

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(model, str(ckpt_dir))
        if torch.backends.mps.is_available():
            model.to("mps")
        elif torch.cuda.is_available():
            model.to("cuda")
        model.eval()

        prompt = tokenizer.apply_chat_template(
            probe_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not raw or raw.count("!") > len(raw) // 2:
            return False
        extract_json_object(raw)
        return True
    except Exception:
        return False


def _pick_checkpoint(
    ckpt_root: Path,
    base_model: str,
    train_rows: list[dict],
) -> Path | None:
    ckpt_dirs = sorted(ckpt_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not ckpt_dirs:
        return None
    probe = train_rows[0]["messages"][:-1] if train_rows else None
    if not probe:
        return ckpt_dirs[-1]
    for ckpt in reversed(ckpt_dirs):
        if _checkpoint_ok(ckpt, base_model, probe):
            print(f"Selected checkpoint: {ckpt.name}", file=sys.stderr)
            return ckpt
    print(f"WARNING: no checkpoint passed probe; using {ckpt_dirs[-1].name}", file=sys.stderr)
    return ckpt_dirs[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-0.5B for OSC intents")
    parser.add_argument("--model", default=os.environ.get("LLM4OSC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--train", type=Path, default=_repo_root() / DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=_repo_root() / DEFAULT_VAL)
    parser.add_argument("--output", type=Path, default=_repo_root() / DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args(argv)

    if not args.train.is_file():
        print(f"ERROR: missing {args.train} — run: python benchmarks/generate_train.py", file=sys.stderr)
        return 1

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(
            "ERROR: training extras not installed. Run: python -m pip install -e '.[train]'",
            file=sys.stderr,
        )
        return 1

    train_rows = _load_jsonl(args.train)
    val_rows = _load_jsonl(args.val) if args.val.is_file() else []

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_cuda = torch.cuda.is_available()
    use_qlora = use_cuda
    device_map = "auto" if use_cuda else None

    if use_qlora:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb,
            device_map=device_map,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.float32
        if torch.backends.mps.is_available():
            dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        if torch.backends.mps.is_available():
            model.to("mps")
        elif use_cuda:
            model.to("cuda")

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = Dataset.from_list([_format_row(tokenizer, r["messages"]) for r in train_rows])
    eval_ds = (
        Dataset.from_list([_format_row(tokenizer, r["messages"]) for r in val_rows])
        if val_rows
        else None
    )

    args.output.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = _repo_root() / "data" / "manifest.json"
    data_hash = None
    if manifest_path.is_file():
        data_hash = json.loads(manifest_path.read_text(encoding="utf-8")).get("data_hash")

    training_args = SFTConfig(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=use_cuda,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        max_length=args.max_seq_len,
        dataset_text_field="text",
        gradient_checkpointing=False,
        dataloader_pin_memory=use_cuda,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print(f"Training on {len(train_rows)} rows ({'QLoRA' if use_qlora else 'LoRA'})...", file=sys.stderr)
    trainer.train()
    save_src = _pick_checkpoint(args.output / "checkpoints", args.model, train_rows)

    import shutil
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if save_src and save_src.is_dir():
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ):
            src = save_src / name
            if src.is_file():
                shutil.copy2(src, adapter_dir / name)
    else:
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    card = {
        "base_model": args.model,
        "adapter_path": str(adapter_dir.relative_to(_repo_root())),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "data_hash": data_hash,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "quantization": "4bit" if use_qlora else "none",
    }
    (args.output / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(f"Saved adapter to {adapter_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
