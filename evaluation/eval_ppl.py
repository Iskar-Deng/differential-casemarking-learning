#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate all GPT-2 checkpoints under CHECKPOINT_PATH/<run_id>
on the combined test set (affected + unaffected + invalid),
compute overall PPL per checkpoint, and save CSV + PNG.

Usage:
    python -m evaluation.eval_ppl --run-id independent_Anone_Pdefinite
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# ✅ 引用 utils 中的路径
from utils import CHECKPOINT_PATH, DATA_PATH


# ----------------------------
# Load text files
# ----------------------------
def load_texts(path: Path) -> List[str]:
    texts = []
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "text" in obj:
                        texts.append(obj["text"])
                except json.JSONDecodeError:
                    continue
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    texts.append(line.strip())
    return texts


# ----------------------------
# Load model + tokenizer
# ----------------------------
def load_model(checkpoint_path: Path, device: str):
    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint_path))
    model = GPT2LMHeadModel.from_pretrained(str(checkpoint_path))
    tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval().to(device)
    return model, tokenizer


# ----------------------------
# Compute NLL for a text
# ----------------------------
@torch.no_grad()
def negative_log_likelihood_for_text(model, tokenizer, text: str, device: str) -> Tuple[float, int]:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    context_length = getattr(model.config, "n_positions", 1024)
    overlap = 50
    stride = max(context_length - overlap, 1)
    nll_total, total_tokens = 0.0, 0
    seq_len = input_ids.size(1)

    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - context_length, 0)
        end_loc = min(i + stride, seq_len)
        input_ids_window = input_ids[:, begin_loc:end_loc]
        trg_len = end_loc - i
        labels = input_ids_window.clone()
        labels[:, :-trg_len] = -100
        outputs = model(input_ids=input_ids_window, labels=labels)
        if trg_len > 0:
            nll_total += outputs.loss.item() * trg_len
            total_tokens += trg_len
        if end_loc == seq_len:
            break

    return float(nll_total), int(total_tokens)


# ----------------------------
# Compute dataset PPL
# ----------------------------
@torch.no_grad()
def dataset_ppl(model, tokenizer, texts: List[str], device: str) -> Dict[str, float]:
    total_nll, total_tokens = 0.0, 0
    for t in tqdm(texts, desc="texts", leave=False, dynamic_ncols=True):
        nll, ntoks = negative_log_likelihood_for_text(model, tokenizer, t, device)
        total_nll += nll
        total_tokens += ntoks
    mean_nll = total_nll / max(total_tokens, 1)
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    return {"ppl": ppl, "mean_nll": mean_nll, "tokens": total_tokens}


# ----------------------------
# Find checkpoints
# ----------------------------
def find_checkpoints(run_dir: Path) -> List[Path]:
    cks = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    cks.sort(key=lambda p: int(p.name.split("-")[-1]))
    return cks


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate combined PPL for all checkpoints by run_id.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID, e.g. independent_Anone_Pdefinite")
    args = parser.parse_args()

    run_id = args.run_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    # === Resolve paths ===
    ckpt_root = Path(CHECKPOINT_PATH) / run_id
    data_root = Path(DATA_PATH) / "perturbed" / run_id
    out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Load and merge test sets ===
    test_paths = [
        data_root / "test_affected.txt",
        data_root / "test_unaffected.txt",
        data_root / "test_invalid.txt",
    ]
    all_texts = []
    for p in test_paths:
        texts = load_texts(p)
        print(f"[Info] Loaded {len(texts)} from {p}")
        all_texts.extend(texts)

    if not all_texts:
        raise RuntimeError("No test texts found.")
    print(f"[Info] Combined test size: {len(all_texts)}")

    # === Find checkpoints ===
    checkpoints = find_checkpoints(ckpt_root)
    print(f"[Info] Found {len(checkpoints)} checkpoints under {ckpt_root}")
    if not checkpoints:
        raise RuntimeError("No checkpoints found.")

    results = []

    for ckpt in tqdm(checkpoints, desc="Evaluating checkpoints", dynamic_ncols=True):
        try:
            step = int(ckpt.name.split("-")[-1])
        except ValueError:
            continue

        try:
            model, tokenizer = load_model(ckpt, device)
        except Exception as e:
            tqdm.write(f"[Warn] Failed to load {ckpt}: {e}")
            continue

        try:
            metrics = dataset_ppl(model, tokenizer, all_texts, device)
            results.append({
                "checkpoint": str(ckpt),
                "step": step,
                "ppl": metrics["ppl"],
                "mean_nll": metrics["mean_nll"],
                "tokens": metrics["tokens"],
            })
        except Exception as e:
            tqdm.write(f"[Warn] Failed {ckpt}: {e}")
            continue

        del model
        torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("No successful evaluations.")

    # === Save CSV ===
    df = pd.DataFrame(results).sort_values("step").reset_index(drop=True)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Info] Saved CSV: {csv_path}")

    # === Plot curve ===
    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["ppl"], marker="o", label="Combined Test Set")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Perplexity (PPL)")
    plt.title(f"Combined Test PPL ({run_id})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = out_dir / "ppl_curve.png"
    plt.savefig(png_path, dpi=200)
    print(f"[Info] Saved plot: {png_path}")

    # === Print best checkpoint ===
    best = df.loc[df["ppl"].idxmin()]
    print(f"\n[Best checkpoint] step={best['step']} ppl={best['ppl']:.4f} ({best['checkpoint']})")


if __name__ == "__main__":
    main()
