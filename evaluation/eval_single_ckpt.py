#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import contextlib

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm.auto import tqdm
from utils import AGENT_MARK, PATIENT_MARK  # 用于识别“带标记”的句子


def load_tokenizer(checkpoint: Path) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint: Path, device: str) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(str(checkpoint)).to(device)
    model.eval()
    return model


def read_minipairs(jsonl_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pairs.append((obj["sentence_good"], obj["sentence_bad"]))
            except Exception:
                continue
    return pairs


def batch_iter(items: List[Tuple[str, str]], batch_size: int) -> Iterable[List[Tuple[str, str]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _masked_labels(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    return labels


@torch.inference_mode()
def per_sample_loss(texts: List[str], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, device: str) -> List[float]:
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

    vocab = shift_logits.size(-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(shift_logits.view(-1, vocab), shift_labels.view(-1))
    token_loss = token_loss.view(shift_labels.size())

    mask = (shift_labels != -100).float()
    denom = mask.sum(dim=-1).clamp_min(1.0)
    sample_loss = (token_loss * mask).sum(dim=-1) / denom

    return sample_loss.detach().float().cpu().tolist()


def evaluate_single_checkpoint(
    checkpoint: Path,
    jsonl_path: Path,
    batch_size: int = 16,
    fp16: bool = False,
    device: str = None,
    save_details: Path = None,
) -> Dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(checkpoint)
    model = load_model(checkpoint, device)

    pairs = read_minipairs(jsonl_path)
    if not pairs:
        raise RuntimeError(f"No valid pairs found in {jsonl_path}")

    correct = 0
    total = 0
    delta_losses = []
    delta_correct = []
    delta_wrong = []

    marked_correct = 0
    marked_total = 0
    unmarked_correct = 0
    unmarked_total = 0

    details: List[Dict] = []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (fp16 and device == "cuda")
        else contextlib.nullcontext()
    )

    with amp_ctx:
        for batch in tqdm(batch_iter(pairs, batch_size), total=(len(pairs) + batch_size - 1) // batch_size, desc="Evaluating"):
            flat_texts: List[str] = []
            for g, b in batch:
                flat_texts.extend([g, b])

            losses = per_sample_loss(flat_texts, model, tokenizer, device)
            for i, (g, b) in enumerate(batch):
                lg = losses[2 * i]
                lb = losses[2 * i + 1]
                diff = lb - lg
                delta_losses.append(diff)

                pred = 1 if lg < lb else 0
                label = 1
                is_correct = (pred == label)

                correct += int(is_correct)
                total += 1
                (delta_correct if is_correct else delta_wrong).append(diff)

                is_marked = (AGENT_MARK in g or PATIENT_MARK in g)
                if is_marked:
                    marked_total += 1
                    if is_correct:
                        marked_correct += 1
                else:
                    unmarked_total += 1
                    if is_correct:
                        unmarked_correct += 1

                if save_details is not None:
                    details.append(
                        {
                            "sentence_good": g,
                            "sentence_bad": b,
                            "loss_good": lg,
                            "loss_bad": lb,
                            "loss_diff": diff,
                            "correct": is_correct,
                            "marked": is_marked,
                        }
                    )

    accuracy = correct / total if total > 0 else float("nan")
    marked_acc = marked_correct / marked_total if marked_total > 0 else float("nan")
    unmarked_acc = unmarked_correct / unmarked_total if unmarked_total > 0 else float("nan")

    if save_details is not None:
        save_details.parent.mkdir(parents=True, exist_ok=True)
        with save_details.open("w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mean_delta": sum(delta_losses) / len(delta_losses) if delta_losses else float("nan"),
        "mean_delta_correct": sum(delta_correct) / len(delta_correct) if delta_correct else float("nan"),
        "mean_delta_wrong": sum(delta_wrong) / len(delta_wrong) if delta_wrong else float("nan"),
        "marked_acc": marked_acc,
        "marked_total": marked_total,
        "unmarked_acc": unmarked_acc,
        "unmarked_total": unmarked_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single GPT-2 checkpoint on minipairs JSONL.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-details", type=Path, default=None)
    args = parser.parse_args()

    res = evaluate_single_checkpoint(
        checkpoint=args.checkpoint,
        jsonl_path=args.jsonl,
        batch_size=args.batch_size,
        fp16=args.fp16,
        device=args.device,
        save_details=args.save_details,
    )

    print("\n===== Evaluation Result =====")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Dataset      : {args.jsonl}")
    print(f"Total        : {res['total']}")
    print(f"Correct      : {res['correct']}")
    print(f"Accuracy     : {res['accuracy']:.4f}")
    print(f"Mean Δloss   : {res['mean_delta']:.4f}")
    print(f"Mean Δ(correct): {res['mean_delta_correct']:.4f}")
    print(f"Mean Δ(wrong)  : {res['mean_delta_wrong']:.4f}")
    print(f"Marked acc   : {res['marked_acc']:.4f} (n={res['marked_total']} / {res['total']}, {res['marked_total']/res['total']:.1%})")
    print(f"Unmarked acc : {res['unmarked_acc']:.4f} (n={res['unmarked_total']} / {res['total']}, {res['unmarked_total']/res['total']:.1%})")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
