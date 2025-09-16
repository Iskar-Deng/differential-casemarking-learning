#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a YAML config from a perturbation output directory.

新增功能:
  --oversample N  将 train_affected.txt 重复采样 N 倍 (默认 1)
  --warmup        只使用 affected/unaffected (不加载 invalid)，用于 warmup 阶段
"""

import argparse
from pathlib import Path
import yaml
from utils import DATA_PATH, CONFIG_PATH, CHECKPOINT_PATH, CACHE_PATH

TEMPLATE = {
    "run_id": "__RUN_ID__",
    "model_name": "gpt2",
    "effective_bsz": 96,
    "seed": 42,
    "block_size": 1024,
    "resume": True,
    "resume_checkpoint": None,
    "checkpoint_frequency": [[50, 500], [100, 1000], [1000, 10000]],
    "artifacts": {
        "cache_dir": "__CACHE_DIR__",
        "run_dir": "__RUN_DIR__",
    },
    "training_arguments": {
        "max_steps": 8000,
        "per_device_train_batch_size": 4,
        "optim": "adamw_torch_fused",
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": False,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 50,
        "eval_steps": 1000,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "prediction_loss_only": True,
        "dataloader_num_workers": 6,
        "dataloader_pin_memory": True,
        "report_to": ["tensorboard"],
    },
    "data": {"train_files": [], "eval_files": []},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True,
                    help="Perturbation output directory")
    ap.add_argument("--eval-split", type=str, choices=["valid", "test"], default="valid")
    ap.add_argument("--oversample", type=int, default=1,
                    help="重复采样 train_affected.txt N 次 (默认 1 表示不重复)")
    ap.add_argument("--warmup", action="store_true",
                    help="仅使用 affected/unaffected 训练集，不加载 invalid")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.input_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    run_id = base_dir.name + ("_warmup" if args.warmup else "")

    def pick_paths(split: str):
        affected = list(base_dir.glob(f"{split}_affected.txt"))
        unaffected = list(base_dir.glob(f"{split}_unaffected.txt"))
        invalid = list(base_dir.glob(f"{split}_invalid.txt"))
        if not affected or not unaffected:
            raise RuntimeError(f"Split '{split}' not found under {base_dir}")
        paths = {
            "affected": str(affected[0]),
            "unaffected": str(unaffected[0]),
        }
        if invalid:
            paths["invalid"] = str(invalid[0])
        return paths

    train_paths = pick_paths("train")
    eval_paths = pick_paths(args.eval_split)

    train_files = []
    if args.oversample > 1:
        train_files.extend([train_paths["affected"]] * args.oversample)
        train_files.extend([train_paths["unaffected"]] * args.oversample)
    else:
        train_files.append(train_paths["affected"])
        train_files.append(train_paths["unaffected"])
    if (not args.warmup) and "invalid" in train_paths:
        train_files.append(train_paths["invalid"])

    eval_files = [eval_paths["affected"], eval_paths["unaffected"]]
    if (not args.warmup) and "invalid" in eval_paths:
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
    print(f"  warmup     : {args.warmup}")
    print(f"  oversample : {args.oversample}x affected")
    print(f"  cache_dir  : {cfg['artifacts']['cache_dir']}")
    print(f"  run_dir    : {cfg['artifacts']['run_dir']}")
    print(f"  train_files ({len(train_files)}):")
    for p in train_files:
        print("   -", p)
    print(f"  eval_files ({len(eval_files)}):")
    for p in eval_files:
        print("   -", p)


if __name__ == "__main__":
    main()
