#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate GPT-2 checkpoints on a JSONL validation set, compute PPL,
and save CSV/TXT results. Uses utils.CHECKPOINT_PATH and utils.DATA_PATH.

Examples:
    python eval_ppl.py --run-id rule_A+P
    python eval_ppl.py --run-id rule_A+P_with_invalid
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from tqdm.auto import tqdm
from utils import CHECKPOINT_PATH, DATA_PATH


# ----------------------------
# Data loading
# ----------------------------
def load_jsonl_texts(path: Path) -> List[str]:
    path = Path(path)
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" in obj:
                texts.append(obj["text"])
    return texts


# ----------------------------
# Model loading
# ----------------------------
def load_model(checkpoint_path: Path, device: str):
    checkpoint_path = Path(checkpoint_path)
    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint_path))
    model = GPT2LMHeadModel.from_pretrained(str(checkpoint_path))
    tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval().to(device)
    return model, tokenizer

@torch.no_grad()
def negative_log_likelihood_for_text(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    text: str,
    device: str,
    show_window_progress: bool = False, 
    position: int = 2,
    leave: bool = False,
    desc: Optional[str] = None,
) -> Tuple[float, int]:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    context_length = getattr(model.config, "n_positions", None)
    if context_length is None:
        context_length = getattr(model.config, "max_position_embeddings", 1024)
    context_length = int(context_length)

    overlap = 50
    stride = max(context_length - overlap, 1)

    nll_total = 0.0
    total_target_tokens = 0

    seq_len = input_ids.size(1)
    indices = list(range(0, seq_len, stride))

    iterator = indices
    if show_window_progress:
        iterator = tqdm(indices, desc=desc or "Windows", unit="win",
                        position=position, leave=leave, dynamic_ncols=True)

    for i in iterator:
        begin_loc = max(i + stride - context_length, 0)  
        end_loc   = min(i + stride, seq_len)             
        input_ids_window = input_ids[:, begin_loc:end_loc]

        trg_len = end_loc - i
        labels = input_ids_window.clone()
        labels[:, :-trg_len] = -100

        outputs = model(input_ids=input_ids_window, labels=labels)
        if trg_len > 0:
            nll_total += outputs.loss.item() * trg_len
            total_target_tokens += trg_len

        if end_loc == seq_len:
            break

    return float(nll_total), int(total_target_tokens)


@torch.no_grad()
def dataset_ppl(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: List[str],
    device: str,
    show_text_progress: bool = False,   
    text_desc: Optional[str] = None,
    position: int = 1,
    leave: bool = False,
    show_window_progress: bool = False, 
) -> Dict[str, float]:
    total_nll = 0.0
    total_tokens = 0

    iterator = texts
    if show_text_progress:
        iterator = tqdm(
            texts,
            desc=text_desc or "Evaluating texts",
            unit="text",
            position=position,
            leave=leave,
            dynamic_ncols=True,
        )

    for t in iterator:
        nll, ntoks = negative_log_likelihood_for_text(
            model, tokenizer, t,
            device=device,
            show_window_progress=show_window_progress,
            position=position + 1,
            leave=False,
            desc="Windows",
        )
        total_nll += nll
        total_tokens += ntoks

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    return {"ppl": ppl, "mean_nll": mean_nll, "tokens": total_tokens}


# ----------------------------
# Checkpoint discovery
# ----------------------------
def find_checkpoints_recursive(root: Path) -> List[Path]:
    root = Path(root)
    if root.is_dir() and root.name.startswith("checkpoint-"):
        cks = [root]
    else:
        cks = [p for p in root.rglob("checkpoint-*") if p.is_dir()]

    def step_key(p: Path):
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return float("inf")

    cks.sort(key=step_key)
    return cks


# ----------------------------
# Path resolver from run-id
# ----------------------------
def resolve_paths_from_run_id(run_id: str) -> Dict[str, Path]:
    use_with_invalid = run_id.endswith("_with_invalid")
    base_name = run_id[:-13] if use_with_invalid else run_id  # 去掉 '_with_invalid'
    subset = "train_with_invalid" if use_with_invalid else "train_without_invalid"

    ckpt_root = Path(CHECKPOINT_PATH) / run_id
    val_path = Path(DATA_PATH) / "perturbed_model" / base_name / subset / "validation.jsonl"

    return {
        "ckpt_root": ckpt_root,
        "val_path": val_path,
        "subset": subset,
        "base_name": base_name,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate PPL for GPT-2 checkpoints on a JSONL set (by run-id).")
    parser.add_argument("--run-id", type=str, required=True,
                        help="e.g., rule_A+P or rule_A+P_with_invalid (controls data subset and checkpoint root).")
    parser.add_argument("--max-checkpoints", type=int, default=None,
                        help="If set, evaluate at most this many checkpoints.")
    parser.add_argument("--show-window-progress", action="store_true",
                        help="Show sliding-window progress (off by default).")
    parser.add_argument("--out-dir", type=Path, default=Path("results"),
                        help="Base output directory. Files will be saved under <out-dir>/<run-id>/")
    parser.add_argument("--no-plot", action="store_true", help="(Deprecated) plotting is disabled by default.")
    args = parser.parse_args()

    paths = resolve_paths_from_run_id(args.run_id)
    ckpt_root: Path = paths["ckpt_root"]
    val_path: Path = paths["val_path"]

    out_dir = args.out_dir / args.run_id      # results/<run-id>/
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Output directory: {out_dir.resolve()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    print(f"[Info] Using device: {device}")
    print(f"[Info] Run ID: {args.run_id}")
    print(f"[Info] Checkpoint root: {ckpt_root}")
    print(f"[Info] Validation file: {val_path}")

    # Load validation texts
    texts = load_jsonl_texts(val_path)
    print(f"[Info] Loaded {len(texts)} validation texts")
    if len(texts) == 0:
        raise RuntimeError("Validation set is empty or path is wrong.")

    # Discover checkpoints
    checkpoints = find_checkpoints_recursive(ckpt_root)
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[: args.max_checkpoints]
    print(f"[Info] Found {len(checkpoints)} checkpoints under {ckpt_root}")
    if len(checkpoints) == 0:
        raise RuntimeError("No checkpoints found. Check folder structure.")

    results: List[Dict] = []

    # Outer: checkpoint-level progress
    for ckpt in tqdm(checkpoints, desc="Evaluating checkpoints", unit="ckpt", position=0, dynamic_ncols=True):
        try:
            model, tokenizer = load_model(ckpt, device)
        except Exception as e:
            tqdm.write(f"[Warn] Failed to load {ckpt}: {e}")
            continue

        # Inner: text-level progress inside each checkpoint
        try:
            metrics = dataset_ppl(
                model, tokenizer, texts,
                device=device,
                show_text_progress=True,
                text_desc=f"texts @ {ckpt.name}",
                position=1,
                leave=False,
                show_window_progress=args.show_window_progress,
            )
        except Exception as e:
            tqdm.write(f"[Warn] Failed to evaluate {ckpt}: {e}")
            continue

        try:
            step = int(ckpt.name.split("-")[-1])
        except Exception:
            step = None

        results.append({
            "checkpoint": str(ckpt),
            "step": step,
            "ppl": metrics["ppl"],
            "mean_nll": metrics["mean_nll"],
            "tokens": metrics["tokens"],
        })

    if not results:
        raise RuntimeError("No successful results. See warnings above.")

    # Save results
    df = pd.DataFrame(results).sort_values(by=["step"], na_position="last").reset_index(drop=True)

    out_csv = out_dir / f"{args.run_id}_results.csv"
    out_txt = out_dir / f"{args.run_id}_results.txt"

    df.to_csv(out_csv, index=False)
    with out_txt.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(f"{r['step']}\t{r['ppl']:.6f}\t{r['mean_nll']:.6f}\t{int(r['tokens'])}\t{r['checkpoint']}\n")

    print(f"[Info] Saved: {out_csv.resolve()}")
    print(f"[Info] Saved: {out_txt.resolve()}")

    # Best checkpoint by PPL
    best = df.loc[df["ppl"].idxmin()]
    print("\n[Best checkpoint by PPL]")
    print(best)


if __name__ == "__main__":
    main()
