#!/usr/bin/env python3
"""
SFT 训练入口，支持命令行参数和 YAML 配置文件。

用法:
    python sft.py --config configs/sft_qwen2.5_1.5b.yaml
    python sft.py --model ./Qwen2.5-1.5B --data ./data/sft_chat_22k.jsonl --output ./output
"""

import argparse
import yaml
from pathlib import Path

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--config", type=str, help="YAML config file (overridden by CLI args)")
    parser.add_argument("--model", type=str, default=".", help="Model path or HuggingFace ID")
    parser.add_argument("--data", type=str, default="./data/sft_chat_22k.jsonl", help="Data file path (JSONL)")
    parser.add_argument("--output", type=str, default="./output/sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--packing", action="store_true", default=True)
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    parser.add_argument("--run_name", type=str, default="sft")
    return parser.parse_args()


def merge_config(args):
    """从 YAML 加载默认配置，CLI 参数覆盖 YAML。"""
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    # CLI 参数优先级高于 YAML（仅当显式传入时覆盖）
    cli_overrides = {k: v for k, v in vars(args).items() if k != "config"}
    for k, v in cli_overrides.items():
        if k == "data":
            cfg.setdefault("data", v)
        elif k == "output":
            cfg.setdefault("output_dir", v)
        elif k == "grad_accum":
            cfg.setdefault("gradient_accumulation_steps", v)
        elif k == "max_length":
            cfg.setdefault("max_seq_length", v)
        else:
            cfg.setdefault(k, v)
    return cfg


def main():
    args = parse_args()
    cfg = merge_config(args)

    print(f"Model:  {cfg['model']}")
    print(f"Data:   {cfg['data']}")
    print(f"Output: {cfg.get('output_dir', cfg.get('output', './output/sft'))}")
    print(f"Config: { {k: v for k, v in cfg.items() if k not in ('data', 'model')} }")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    dataset = load_dataset("json", data_files=cfg["data"], split="train")
    print(f"Dataset size: {len(dataset)}")

    # Training args
    training_args = SFTConfig(
        output_dir=cfg.get("output_dir", cfg.get("output", "./output/sft")),
        per_device_train_batch_size=cfg.get("batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=True,
        num_train_epochs=cfg.get("epochs", 3),
        learning_rate=cfg.get("lr", 2e-5),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        max_seq_length=cfg.get("max_seq_length", 2048),
        packing=cfg.get("packing", True),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to=cfg.get("report_to", "wandb"),
        run_name=cfg.get("run_name", "sft"),
    )

    trainer = SFTTrainer(
        model=cfg["model"],
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
