#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate YAML config (pure declarative, no sampling or I/O)
-----------------------------------------------------------
仅生成配置文件，不采样、不创建临时文件。

训练阶段再根据 oversample 值执行采样逻辑。
"""

import argparse
import yaml
from pathlib import Path
from utils import CONFIG_PATH, CHECKPOINT_PATH, CACHE_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--oversample", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.input_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} 不存在")

    run_id = base_dir.name
    print(f"[RunID] {run_id}")

    # 定义基础路径（不访问文件）
    def get_paths(split: str):
        return {
            "affected": str(base_dir / f"{split}_affected.txt"),
            "unaffected": str(base_dir / f"{split}_unaffected.txt"),
            "invalid": str(base_dir / f"{split}_invalid.txt"),
        }

    train_paths = get_paths("train")
    eval_paths = get_paths("valid")

    cfg = {
        "run_id": run_id,
        "model_name": "gpt2",
        "seed": 42,
        "block_size": 256,
        "resume": True,
        "resume_checkpoint": None,
        "checkpoint_frequency": [
            [1000, 10000],
            [5000, 100000],
            [10000, 300000],
        ],
        "artifacts": {
            "cache_dir": str(Path(CACHE_PATH) / run_id),
            "run_dir": str(Path(CHECKPOINT_PATH) / run_id),
        },
        "training_arguments": {
            "max_steps": 300000,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 16,
            "per_device_eval_batch_size": 2,
            "optim": "adamw_torch",
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": True,
            "learning_rate": 2.0e-05,
            "weight_decay": 0.01,
            "warmup_steps": 0,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "logging_steps": 50,
            "eval_steps": 0,
            "eval_strategy": "no",
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "prediction_loss_only": True,
            "dataloader_num_workers": 2,
            "dataloader_pin_memory": True,
            "max_grad_norm": 0.5,
            "report_to": ["tensorboard"],
        },
        "data": {
            "train_files": [
                train_paths["affected"],
                train_paths["unaffected"],
                train_paths["invalid"],
            ],
            "eval_files": [
                eval_paths["affected"],
                eval_paths["unaffected"],
                eval_paths["invalid"],
            ],
            "oversample_plan": {
                "affected": args.oversample,
                "unaffected": args.oversample,
            },
        },
    }

    out_dir = Path(CONFIG_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / f"{run_id}.yaml"
    if out_yaml.exists() and not args.overwrite:
        raise FileExistsError(f"{out_yaml} 已存在。使用 --overwrite 覆盖。")

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"\n✅ [OK] 写入配置文件: {out_yaml}")
    print(f"  oversample = {args.oversample}x (仅记录，不采样)")
    print("=====================================")
    print("train_files:")
    print(" -", train_paths["affected"])
    print(" -", train_paths["unaffected"])
    print(" -", train_paths["invalid"])


if __name__ == "__main__":
    main()
