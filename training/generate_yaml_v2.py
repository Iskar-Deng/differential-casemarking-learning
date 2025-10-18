#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_config_static_v5.py (Balanced version)
------------------------------------------------
自动计算采样比例，使：
  1. affected 与 unaffected 同步放大（比例不变）
  2. affected 占整体语料比例保持 15.55%
  3. 总样本数量保持不变
  4. invalid 相应缩减

功能：
- 自动读取 train_affected / train_unaffected / train_invalid 的行数
- 自动计算放大倍数 r 与缩减倍数 s
- 短句合并 + 静态 oversample / undersample
- 输出 YAML 配置文件（与既有格式一致）
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
    """将过短句子合并为较长行，提高训练效率。"""
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
    """读取文件并生成静态 oversample/undersample 版本。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]
    if not lines:
        print(f"[Warn] 文件为空：{path}")
        return path

    n = len(lines)
    if ratio <= 0:
        print(f"[Error] 非法采样倍数: {ratio}")
        return path

    # oversample
    if ratio > 1.0:
        n_full = int(ratio)
        frac = ratio - n_full
        oversampled = lines * n_full
        if frac > 1e-6:
            random.seed(seed)
            n_take = max(1, int(math.ceil(n * frac)))
            oversampled += random.sample(lines, min(n_take, n))
    # undersample
    elif ratio < 1.0:
        random.seed(seed)
        n_take = max(1, int(math.floor(n * ratio)))
        oversampled = random.sample(lines, n_take)
    else:
        oversampled = lines

    out_path = path.with_name(f"{path.stem}_x{ratio:.2f}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for line in oversampled:
            f.write(line + "\n")

    print(f"[Resample] {path.name}: {n} → {len(oversampled)} lines ({ratio:.2f}x) → {out_path.name}")
    return out_path


def count_lines(path: Path) -> int:
    """快速统计文件行数"""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())


# =====================================================
# New sampling ratio calculator
# =====================================================

def auto_calc_balanced_ratios(base_dir: Path, target_ratio: float = 0.1555):
    """
    自动计算双向采样比例：
      - affected/unaffected 同步放大倍数 r
      - invalid 缩小倍数 s
      - 保持总样本数不变且 affected 占比≈target_ratio
    """
    A = count_lines(base_dir / "train_affected.txt")
    U = count_lines(base_dir / "train_unaffected.txt")
    I = count_lines(base_dir / "train_invalid.txt")

    total = A + U + I
    p = target_ratio

    num = total * p * I
    denom = (A + U) * (p * (I + U) - A * (1 - p))

    if denom <= 0:
        print(f"[Error] 无法计算倍率 (denom={denom:.4f})，请检查数据分布。")
        return 1.0, 1.0

    r = num / denom
    s = (total - r * (A + U)) / I

    print(f"[Auto balance] A={A}, U={U}, I={I} → p={p:.4f}")
    print(f" → 扩大倍数 r={r:.3f}, 缩小倍数 s={s:.3f}")
    print(f" → 校验: 新A占比={(A*r)/(A*r+U*r+I*s):.4%}, 总行数={A*r+U*r+I*s:.0f}")

    return r, s


# =====================================================
# Config builder
# =====================================================

def build_cfg(base_dir: Path, over_r: float, shrink_s: float):
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
        "affected": static_oversample(merged_train_paths["affected"], over_r),
        "unaffected": static_oversample(merged_train_paths["unaffected"], over_r),
        "invalid": static_oversample(merged_train_paths["invalid"], shrink_s),
    }

    cfg = {
        "run_id": run_id,
        "model_name": "gpt2",
        "seed": 42,
        "block_size": 1024,
        "resume": True,
        "resume_checkpoint": None,
        "checkpoint_frequency": [
            [1000, 20000],
            [5000, 100000],
        ],
        "artifacts": {
            "cache_dir": f"/workspace/differential-casemarking-learning/cache/{run_id}",
            "run_dir": f"/workspace/differential-casemarking-learning/checkpoints/{run_id}",
        },
        "training_arguments": {
            "max_steps": 70000,
            "learning_rate": 2.0e-05,
            "weight_decay": 0.01,
            "warmup_steps": 0,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "per_device_train_batch_size": 48,
            "gradient_accumulation_steps": 2,
            "per_device_eval_batch_size": 64,
            "optim": "adamw_torch",
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 2,
            "dataloader_pin_memory": True,
            "max_grad_norm": 0.5,
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

    print(f"[Config] batch={cfg['training_arguments']['per_device_train_batch_size']} × "
          f"grad_acc={cfg['training_arguments']['gradient_accumulation_steps']} "
          f"(effective={cfg['training_arguments']['per_device_train_batch_size'] * cfg['training_arguments']['gradient_accumulation_steps']})")

    return cfg


# =====================================================
# Main
# =====================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="perturbed 数据目录")
    ap.add_argument("--oversample", type=float, default=None, help="静态 oversample 倍数 (若指定则 invalid 不缩减)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.input_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} 不存在")

    # 自动计算倍率
    if args.oversample is None:
        over_r, shrink_s = auto_calc_balanced_ratios(base_dir, target_ratio=0.1555)
    else:
        over_r, shrink_s = args.oversample, 1.0
        print(f"[Manual oversample] 使用指定倍数: {over_r}x, invalid 不缩减")

    cfg = build_cfg(base_dir, over_r, shrink_s)

    out_dir = Path(CONFIG_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / f"{base_dir.name}.yaml"

    if out_yaml.exists() and not args.overwrite:
        raise FileExistsError(f"{out_yaml} 已存在。使用 --overwrite 覆盖。")

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"\n✅ [OK] 写入配置文件: {out_yaml}")
    print("=====================================")
    print("train_files:")
    for p in cfg["data"]["train_files"]:
        print(" -", p)
    print("eval_files:")
    for p in cfg["data"]["eval_files"]:
        print(" -", p)


if __name__ == "__main__":
    main()
