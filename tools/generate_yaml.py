#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a YAML config from perturbed_model directory by scanning files.

Example:
  python -m tools.generate_yaml --mode full --strategy A+P --overwrite
  python -m tools.generate_yaml --mode rule --strategy A+P --eval-split test --overwrite
"""
import argparse
from pathlib import Path
import yaml
from utils import DATA_PATH, CONFIG_PATH, CHECKPOINT_PATH, CACHE_PATH

# -------- L4(24GB) 优化默认值 --------
TEMPLATE = {
    "run_id": "__RUN_ID__",
    "model_name": "gpt2",
    "effective_bsz": 96,
    "seed": 42,
    "block_size": 1024,
    "resume": True,
    "resume_checkpoint": None,
    "checkpoint_frequency": [[50, 500],[100, 1000],[1000, 10000]],
    "artifacts": {
        "cache_dir": "__CACHE_DIR__",
        "run_dir": "__RUN_DIR__",
    },
    "training_arguments": {
        "max_steps": 4000,
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
        "save_steps": 1000,
        "save_total_limit": 3,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
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
    ap.add_argument("--mode", type=str, required=True)
    ap.add_argument("--strategy", type=str, required=True)
    ap.add_argument("--with-invalid", action="store_true")
    ap.add_argument("--eval-split", type=str, choices=["valid", "test"], default="valid")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_id = f"{args.mode}_{args.strategy}"
    run_id = f"{dataset_id}_with_invalid" if args.with_invalid else dataset_id
    base_dir = Path(DATA_PATH) / "perturbed_model" / dataset_id

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    def pick_paths(split: str):
        affected = list(base_dir.glob(f"*_{split}_affected.txt"))
        unaffected = list(base_dir.glob(f"*_{split}_unaffected.txt"))
        invalid = list(base_dir.glob(f"*_{split}_invalid.txt"))
        if not affected or not unaffected:
            raise RuntimeError(f"Split '{split}' not found under {base_dir}")
        paths = {
            "affected": str(affected[0]),
            "unaffected": str(unaffected[0]),
        }
        if invalid:
            paths["invalid"] = str(invalid[0])
        return paths

    # train 必须有
    train_paths = pick_paths("train")
    eval_paths  = pick_paths(args.eval_split)

    train_files = [train_paths["affected"], train_paths["unaffected"]]
    eval_files  = [eval_paths["affected"],  eval_paths["unaffected"]]
    if args.with_invalid:
        if "invalid" in train_paths: train_files.append(train_paths["invalid"])
        if "invalid" in eval_paths:  eval_files.append(eval_paths["invalid"])

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
    print(f"  train_files ({len(train_files)}):")
    [print("   -", p) for p in train_files]
    print(f"  eval_files  ({len(eval_files)}):")
    [print("   -", p) for p in eval_files]


if __name__ == "__main__":
    main()
