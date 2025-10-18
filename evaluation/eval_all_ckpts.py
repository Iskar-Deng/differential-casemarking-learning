#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量评估一个文件夹下的所有 GPT-2 checkpoint，并绘制 accuracy 曲线。

Usage:
    python -m evaluation.eval_all_ckpts \
        --run-dir /workspace/differential-casemarking-learning/checkpoints/independent_Anone_Pdefinite \
        --jsonl /workspace/differential-casemarking-learning/evaluation/minimal_pairs/independent_Anone_Pdefinite/valid_test_minimal_pairs.jsonl \
        --out results/independent_Anone_Pdefinite_eval.csv \
        --fp16
"""

import argparse
import json
from pathlib import Path
import contextlib
import re
import csv

import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file as safe_load
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from tqdm.auto import tqdm


# =====================================================
# 基础加载函数
# =====================================================
def load_tokenizer(checkpoint: Path) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint: Path, device: str) -> GPT2LMHeadModel:
    """兼容 safetensors / .bin / 分片 + 修复 _orig_mod. 权重前缀"""
    bin_path = checkpoint / "pytorch_model.bin"
    safe_path = checkpoint / "model.safetensors"
    index_path = checkpoint / "pytorch_model.bin.index.json"

    if safe_path.exists():
        print(f"[Info] 检测到 safetensors 权重文件: {safe_path.name}")
        state_dict = safe_load(str(safe_path))
    elif bin_path.exists():
        print(f"[Info] 检测到 pytorch_model.bin 权重文件")
        state_dict = torch.load(bin_path, map_location="cpu")
    elif index_path.exists():
        print(f"[Info] 检测到分片权重 index 文件: {index_path.name}")
        state_dict = None
    else:
        raise FileNotFoundError(f"未找到权重文件于 {checkpoint}")

    has_orig_mod = state_dict is not None and any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if has_orig_mod:
        print(f"[Fix] 修复 '_orig_mod.' 前缀 ...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        config = GPT2Config.from_pretrained(checkpoint)
        model = GPT2LMHeadModel(config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[Warn] Missing={len(missing)} Unexpected={len(unexpected)}")
    else:
        model = GPT2LMHeadModel.from_pretrained(str(checkpoint), state_dict=state_dict)

    model.to(device)
    model.eval()
    return model


# =====================================================
# 计算每个样本的 loss
# =====================================================
def _masked_labels(inputs):
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    return labels


@torch.inference_mode()
def per_sample_loss(texts, model, tokenizer, device):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(model.config, "n_positions", tokenizer.model_max_length),
    ).to(device)

    labels = _masked_labels(enc)
    outputs = model(**enc, labels=labels)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_loss = token_loss.view(shift_labels.size())

    mask = (shift_labels != -100).float()
    sample_loss = (token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    return sample_loss.detach().cpu().tolist()


# =====================================================
# 评估函数
# =====================================================
def evaluate_checkpoint(checkpoint, jsonl_path, device, batch_size=16, fp16=False):
    tokenizer = load_tokenizer(checkpoint)
    model = load_model(checkpoint, device)

    pairs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["sentence_good"], obj["sentence_bad"]))
    if not pairs:
        raise RuntimeError(f"No pairs in {jsonl_path}")

    correct, total = 0, 0
    delta_losses, delta_correct, delta_wrong = [], [], []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if fp16 and device == "cuda"
        else contextlib.nullcontext()
    )

    with amp_ctx:
        for i in tqdm(range(0, len(pairs), batch_size), desc=f"Eval {checkpoint.name}", leave=False):
            batch = pairs[i : i + batch_size]
            texts = [x for pair in batch for x in pair]
            losses = per_sample_loss(texts, model, tokenizer, device)
            for j, (good, bad) in enumerate(batch):
                lg, lb = losses[2 * j], losses[2 * j + 1]
                diff = lb - lg
                correct_flag = lg < lb
                correct += correct_flag
                total += 1
                delta_losses.append(diff)
                (delta_correct if correct_flag else delta_wrong).append(diff)

    return {
        "step": int(re.findall(r"checkpoint-(\d+)", checkpoint.name)[0]),
        "accuracy": correct / total,
        "mean_delta": sum(delta_losses) / len(delta_losses),
        "mean_delta_correct": sum(delta_correct) / len(delta_correct) if delta_correct else 0.0,
        "mean_delta_wrong": sum(delta_wrong) / len(delta_wrong) if delta_wrong else 0.0,
    }


# =====================================================
# 主函数：批量评估 & 画图
# =====================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="包含 checkpoint-* 子目录的路径")
    ap.add_argument("--jsonl", type=Path, required=True, help="JSONL 格式的 sentence_good/sentence_bad 数据")
    ap.add_argument("--out", type=Path, default=Path("results.csv"), help="输出 CSV 文件路径")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = sorted(
        [p for p in args.run_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda x: int(re.findall(r"checkpoint-(\d+)", x.name)[0])
    )

    results = []
    for ckpt in checkpoints:
        try:
            res = evaluate_checkpoint(ckpt, args.jsonl, device, args.batch_size, args.fp16)
            results.append(res)
            print(f"[OK] step={res['step']:>6d} | acc={res['accuracy']:.3f} | Δ={res['mean_delta']:.3f}")
        except Exception as e:
            print(f"[Skip] {ckpt.name}: {e}")

    # ---- 保存 CSV ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "accuracy", "mean_delta", "mean_delta_correct", "mean_delta_wrong"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[Saved] {args.out}")

    # ---- 绘制曲线 ----
    if results:
        results.sort(key=lambda x: x["step"])
        steps = [r["step"] for r in results]
        accs = [r["accuracy"] for r in results]

        plt.figure(figsize=(7, 4))
        plt.plot(steps, accs, marker="o")
        plt.title(f"Accuracy Curve ({args.run_dir.name})")
        plt.xlabel("Checkpoint Step")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_img = args.out.with_suffix(".png")
        plt.savefig(out_img)
        print(f"[Plot saved] {out_img}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
