#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a YAML config from perturbation summary JSON, writing to CONFIG_PATH.

Usage:
  # 推荐：valid 做评测
  python -m tools.generate_yaml --mode rule --strategy A+P --overwrite

  # 指定评测 split：valid / test
  python -m tools.generate_yaml --mode rule --strategy A+P --eval-split valid --overwrite
  python -m tools.generate_yaml --mode rule --strategy A+P --eval-split test  --overwrite

  # 含 invalid（train/eval 都包含 *_invalid.txt）
  python -m tools.generate_yaml --mode rule --strategy A+P --with-invalid --eval-split valid --overwrite
"""
import argparse
import json
from pathlib import Path
import yaml

from utils import DATA_PATH, CONFIG_PATH, CHECKPOINT_PATH, CACHE_PATH

# -------- L4(24GB) 优化默认值 --------
TEMPLATE = {
    "run_id": "__RUN_ID__",
    "model_name": "gpt2",

    # 让训练脚本用它来推导 GAS：gas = ceil(effective_bsz / per_device_train_batch_size)
    "effective_bsz": 96,         # 可按需调到 128（稳定后再加）
    "seed": 42,
    "block_size": 1024,          # GPT-2 支持到 1024；OOM 再退回 512

    "resume": True,
    "resume_checkpoint": None,

    # 方便你留档；训练脚本不直接用它
    "checkpoint_frequency": [
        [100, 1000],
        [500, 5000],
        [1000, 10000],
    ],

    "artifacts": {
        "cache_dir": "__CACHE_DIR__",  # 将被替换为绝对路径 CACHE_PATH/run_id
        "run_dir": "__RUN_DIR__",      # 将被替换为绝对路径 CHECKPOINT_PATH/run_id
    },

    "training_arguments": {
        # 批量与步数
        "max_steps": 8000,
        "per_device_train_batch_size": 8,  # 24GB L4 通常够用；OOM 就降到 6/4

        # 优化器/精度/显存
        "optim": "adamw_torch_fused",
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": False,   # 先关以换速度；显存吃紧再打开

        # 学习率计划
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",

        # 评测/保存/日志
        "logging_steps": 50,
        "eval_steps": 1000,
        "save_steps": 1000,
        "save_total_limit": 3,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "prediction_loss_only": True,

        # DataLoader
        "dataloader_num_workers": 6,
        "dataloader_pin_memory": True,

        # 写到 TensorBoard（如不需要可改为 []）
        "report_to": ["tensorboard"],
    },

    "data": {
        "train_files": [],
        "eval_files": [],
    },
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True)
    ap.add_argument("--strategy", type=str, required=True)
    ap.add_argument("--with-invalid", action="store_true")
    ap.add_argument("--eval-split", type=str, choices=["valid", "test"], default="valid")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_id = f"{args.mode}_{args.strategy}"
    run_id = f"{dataset_id}_with_invalid" if args.with_invalid else dataset_id

    summary_path = Path(DATA_PATH) / "perturbed_model" / dataset_id / f"summary_{dataset_id}.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    def pick_paths(split: str):
        for item in summary["files"]:
            if item.get("split") == split:
                return item["paths"]
        raise RuntimeError(f"Split '{split}' not found in summary {summary_path}")

    train_paths = pick_paths("train")
    eval_paths  = pick_paths(args.eval_split)

    train_files = [train_paths["affected"], train_paths["unaffected"]]
    eval_files  = [eval_paths["affected"],  eval_paths["unaffected"]]
    if args.with_invalid:
        train_files.append(train_paths["invalid"])
        eval_files.append(eval_paths["invalid"])

    cfg = dict(TEMPLATE)
    cfg["run_id"] = run_id
    cfg["artifacts"] = {
        "cache_dir": str(Path(CACHE_PATH) / run_id),
        "run_dir": str(Path(CHECKPOINT_PATH) / run_id),
    }
    cfg["data"] = {"train_files": train_files, "eval_files": eval_files}

    out_dir = Path(CONFIG_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / f"{run_id}.yaml"

    if out_yaml.exists() and not args.overwrite:
        raise FileExistsError(f"{out_yaml} already exists. Use --overwrite to replace it.")

    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] wrote config: {out_yaml}")
    print(f"  run_id     : {run_id}")
    print(f"  eval_split : {args.eval_split}")
    print(f"  cache_dir  : {cfg['artifacts']['cache_dir']}")
    print(f"  run_dir    : {cfg['artifacts']['run_dir']}")
    print(f"  train_files ({len(train_files)}):"); [print("   -", p) for p in train_files]
    print(f"  eval_files  ({len(eval_files)}):");  [print("   -", p) for p in eval_files]

if __name__ == "__main__":
    main()
