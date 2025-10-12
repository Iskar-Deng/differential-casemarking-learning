#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_config_static_v2.py
------------------------------------------------
静态 oversample + 生成最终 YAML 配置（无 oversample_plan）

- 自动生成 train_affected_x{ratio}.txt / train_unaffected_x{ratio}.txt
- 参数基于最新实验配置（block_size=512, fp16=True, no checkpointing）
"""

import argparse
import yaml
import math
import random
from pathlib import Path
from utils import CONFIG_PATH, CHECKPOINT_PATH, CACHE_PATH


# =====================================================
# Helpers
# =====================================================

def static_oversample(path: Path, ratio: float, seed: int = 42) -> Path:
    """读取文件并生成静态 oversample 版本"""
    if ratio <= 1.0:
        return path

    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]

    if not lines:
        print(f"[Warn] 文件为空：{path}")
        return path

    n = len(lines)
    n_full = int(ratio)
    frac = ratio - n_full

    oversampled = lines * n_full
    if frac > 1e-6:
        random.seed(seed)
        n_take = max(1, int(math.ceil(n * frac)))
        sampled = random.sample(lines, n_take)
        oversampled.extend(sampled)

    out_path = path.with_name(f"{path.stem}_x{ratio}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for line in oversampled:
            f.write(line + "\n")

    print(f"[OK] {path.name}: {n} → {len(oversampled)} lines ({ratio}x) → {out_path.name}")
    return out_path


# =====================================================
# Config builder
# =====================================================

def build_cfg(base_dir: Path, oversample: float):
    run_id = base_dir.name

    def get_paths(split: str):
        return {
            "affected": base_dir / f"{split}_affected.txt",
            "unaffected": base_dir / f"{split}_unaffected.txt",
            "invalid": base_dir / f"{split}_invalid.txt",
        }

    train_paths = get_paths("train")
    eval_paths = get_paths("valid")

    # === 静态 oversample ===
    new_train_paths = {
        "affected": static_oversample(train_paths["affected"], oversample),
        "unaffected": static_oversample(train_paths["unaffected"], oversample),
        "invalid": train_paths["invalid"],
    }

    # === YAML config ===
    cfg = {
        "run_id": run_id,
        "model_name": "gpt2",
        "seed": 42,
        "block_size": 512,
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
            "max_steps": 150000,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 8,
            "per_device_eval_batch_size": 2,
            "optim": "adamw_torch",
            "bf16": False,
            "fp16": True,
            "gradient_checkpointing": False,
            "learning_rate": 2.0e-05,
            "weight_decay": 0.01,
            "warmup_steps": 0,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "logging_steps": 50,
            "eval_strategy": "no",
            "eval_steps": 0,
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "prediction_loss_only": True,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": True,
            "max_grad_norm": 0.5,
            "report_to": ["tensorboard"],
        },
        "data": {
            "train_files": [
                str(new_train_paths["affected"]),
                str(new_train_paths["unaffected"]),
                str(new_train_paths["invalid"]),
            ],
            "eval_files": [
                str(eval_paths["affected"]),
                str(eval_paths["unaffected"]),
                str(eval_paths["invalid"]),
            ],
        },
    }

    return cfg


# =====================================================
# Main
# =====================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="perturbed 数据目录")
    ap.add_argument("--oversample", type=float, default=1.0, help="静态 oversample 倍数")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.input_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} 不存在")

    run_id = base_dir.name
    cfg = build_cfg(base_dir, args.oversample)

    out_dir = Path(CONFIG_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / f"{run_id}.yaml"

    if out_yaml.exists() and not args.overwrite:
        raise FileExistsError(f"{out_yaml} 已存在。使用 --overwrite 覆盖。")

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"\n✅ [OK] 写入配置文件: {out_yaml}")
    print(f"  oversample = {args.oversample}x (静态复制, 无 oversample_plan)")
    print("=====================================")
    print("train_files:")
    for path in cfg["data"]["train_files"]:
        print(" -", path)
    print("eval_files:")
    for path in cfg["data"]["eval_files"]:
        print(" -", path)


if __name__ == "__main__":
    main()
