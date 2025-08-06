#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from utils import CHECKPOINT_PATH, DATA_PATH

def load_model(checkpoint_path: Path, device: str):
    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint_path))
    model = GPT2LMHeadModel.from_pretrained(str(checkpoint_path)).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

@torch.no_grad()
def calculate_loss(sentence: str, model, tokenizer, device: str) -> float:
    inputs = tokenizer(sentence, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()

def find_checkpoints_recursive(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.rglob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1])
    )

def evaluate_checkpoint(model, tokenizer, jsonl_path: Path, device: str) -> Dict:
    preds, labels = [], []
    detailed = []

    with jsonl_path.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating minipairs"):
            item = json.loads(line)
            good = item["sentence_good"]
            bad = item["sentence_bad"]

            loss_good = calculate_loss(good, model, tokenizer, device)
            loss_bad = calculate_loss(bad, model, tokenizer, device)

            pred = 1 if loss_good < loss_bad else 0
            label = 1  # gold: "good" preferred

            preds.append(pred)
            labels.append(label)

            detailed.append({
                "sentence_good": good,
                "sentence_bad": bad,
                "loss_good": loss_good,
                "loss_bad": loss_bad,
                "loss_diff": loss_bad - loss_good,
                "correct": pred == label
            })

    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "correct": sum(preds[i] == labels[i] for i in range(len(labels))),
        "total": len(labels),
        "details": detailed
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True,
                        help="Run ID like rule_A+P or rule_A+P_with_invalid")
    parser.add_argument("--jsonl", type=Path, required=True,
                        help="JSONL file with sentence_good / sentence_bad")
    parser.add_argument("--out-dir", type=Path, default=Path("results"),
                        help="Where to save CSV and details")
    args = parser.parse_args()

    ckpt_root = Path(CHECKPOINT_PATH) / args.run_id
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    checkpoints = find_checkpoints_recursive(ckpt_root)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found under {ckpt_root}")

    all_rows = []
    detailed_all = []

    for ckpt in tqdm(checkpoints, desc="Checkpoints"):
        step = int(ckpt.name.split("-")[-1])
        try:
            model, tokenizer = load_model(ckpt, device)
        except Exception as e:
            tqdm.write(f"[Warn] Failed to load {ckpt}: {e}")
            continue

        result = evaluate_checkpoint(model, tokenizer, args.jsonl, device)

        all_rows.append({
            "step": step,
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "checkpoint": str(ckpt),
        })

        for d in result["details"]:
            d["step"] = step
            d["checkpoint"] = str(ckpt)
            detailed_all.append(d)

    # Save accuracy per checkpoint
    df = pd.DataFrame(all_rows).sort_values("step")
    acc_csv = out_dir / f"{args.run_id}_minipair_results.csv"
    df.to_csv(acc_csv, index=False)
    print(f"[Info] Saved accuracy: {acc_csv.resolve()}")

    # Save detailed info
    details_path = out_dir / f"{args.run_id}_minipair_detailed.jsonl"
    with details_path.open("w", encoding="utf-8") as f:
        for item in detailed_all:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[Info] Saved details: {details_path.resolve()}")

    # Print best checkpoint
    best = df.loc[df["accuracy"].idxmax()]
    print("\nBest Checkpoint:")
    print(best)

if __name__ == "__main__":
    main()
