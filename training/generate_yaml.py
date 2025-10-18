#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_config_static_v3.py (A100-80GB 极致速度版)
------------------------------------------------
功能:
1. 自动合并短句（缓解 I/O 瓶颈，提升 tokenization 速度）
2. 静态 oversample
3. 生成 YAML 训练配置（针对 RunPod A100-80GB 高吞吐环境）

主要优化:
✅ 合并短句 → 每行约 1000 字符
✅ batch=96 × grad_acc=2 → 吃满显存 (~75GB)
✅ 关闭 gradient checkpointing → 最大化算力利用
✅ 启用 bf16 原生加速
✅ 稀疏 checkpoint 频率，减少 I/O 干扰
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

def merge_short_lines(path: Path, max_chars: int = 1000) -> Path:
    """将过短的句子行合并成较长块，提高训练效率。"""
    merged_path = path.with_name(f"{path.stem}_merged.txt")
    with open(path, "r", encoding="utf-8") as f, open(merged_path, "w", encoding="utf-8") as out:
        buf, length = [], 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if length + len(line) > max_chars:
                out.write(" ".join(buf) + "\n")
                buf, length = [line], len(line)
            else:
                buf.append(line)
                length += len(line)
        if buf:
            out.write(" ".join(buf) + "\n")
    print(f"[Merge] {path.name} → {merged_path.name}")
    return merged_path


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
    print(f"[Oversample] {path.name}: {n} → {len(oversampled)} lines ({ratio}x) → {out_path.name}")
    return out_path


# =====================================================
# Config builder (A100 极致版)
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

    merged_train_paths = {k: merge_short_lines(v) for k, v in train_paths.items()}
    merged_eval_paths = {k: merge_short_lines(v) for k, v in eval_paths.items()}

    new_train_paths = {
        "affected": static_oversample(merged_train_paths["affected"], oversample),
        "unaffected": static_oversample(merged_train_paths["unaffected"], oversample),
        "invalid": merged_train_paths["invalid"],
    }

    cfg = {
        "run_id": run_id,
        "model_name": "gpt2",
        "seed": 42,
        "block_size": 1024,
        "resume": True,
        "resume_checkpoint": None,
        "checkpoint_frequency": [
            [500, 10000],
            [1000, 50000],
            [10000, 200000],
        ],
        "artifacts": {
            "cache_dir": f"/workspace/differential-casemarking-learning/cache/{run_id}",
            "run_dir": f"/workspace/differential-casemarking-learning/checkpoints/{run_id}",
        },
        "training_arguments": {
            "max_steps": 150000,
            "learning_rate": 2.0e-05,
            "weight_decay": 0.01,
            "warmup_steps": 0,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",

            # === 🚀 A100 极致速度参数 ===
            "per_device_train_batch_size": 96,
            "gradient_accumulation_steps": 2,
            "per_device_eval_batch_size": 64,
            "optim": "adamw_torch",
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 2,
            "dataloader_pin_memory": True,
            "max_grad_norm": 0.5,

            # === 日志与保存策略 ===
            "logging_steps": 50,
            "eval_strategy": "no",
            "eval_steps": 0,
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "prediction_loss_only": True,
            "remove_unused_columns": False,
            "report_to": ["tensorboard"],
        },
        "data": {
            "train_files": [
                str(new_train_paths["affected"]),
                str(new_train_paths["unaffected"]),
                str(new_train_paths["invalid"]),
            ],
            "eval_files": [
                str(merged_eval_paths["affected"]),
                str(merged_eval_paths["unaffected"]),
                str(merged_eval_paths["invalid"]),
            ],
        },
    }

    print(f"[A100 Config] batch={cfg['training_arguments']['per_device_train_batch_size']} × "
          f"grad_acc={cfg['training_arguments']['gradient_accumulation_steps']} "
          f"(effective={cfg['training_arguments']['per_device_train_batch_size'] * cfg['training_arguments']['gradient_accumulation_steps']})")

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
    print(f"  oversample = {args.oversample}x (短句合并 + 静态复制)")
    print("=====================================")
    print("train_files:")
    for path in cfg["data"]["train_files"]:
        print(" -", path)
    print("eval_files:")
    for path in cfg["data"]["eval_files"]:
        print(" -", path)


if __name__ == "__main__":
    main()
