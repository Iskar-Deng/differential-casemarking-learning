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


def load_tokenizer(checkpoint: Path) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint))
    # GPT-2 默认无 pad_token，这里设置为 eos 以便 padding
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
                # 跳过坏行
                continue
    return pairs


def batch_iter(items: List[Tuple[str, str]], batch_size: int) -> Iterable[List[Tuple[str, str]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _masked_labels(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100  # 屏蔽 padding
    return labels


@torch.inference_mode()
def per_sample_loss(
    texts: List[str],
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: str,
) -> List[float]:
    """对一批文本计算逐样本平均 token 负对数似然 (cross-entropy)。"""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(model.config, "n_positions", tokenizer.model_max_length),
    ).to(device)

    labels = _masked_labels(enc)
    outputs = model(**enc, labels=labels)

    # 手动按样本聚合：shift 以对齐下一个 token 预测
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab = shift_logits.size(-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # [B, T] 逐 token 损失
    token_loss = loss_fct(shift_logits.view(-1, vocab), shift_labels.view(-1))
    token_loss = token_loss.view(shift_labels.size())

    mask = (shift_labels != -100).float()
    # 避免除以 0
    denom = mask.sum(dim=-1).clamp_min(1.0)
    sample_loss = (token_loss * mask).sum(dim=-1) / denom  # [B]

    return sample_loss.detach().float().cpu().tolist()


def evaluate_single_checkpoint(
    checkpoint: Path,
    jsonl_path: Path,
    batch_size: int = 16,
    fp16: bool = False,
    device: str = None,
    save_details: Path = None,
) -> Dict:
    # 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载
    tokenizer = load_tokenizer(checkpoint)
    model = load_model(checkpoint, device)

    # 读取数据
    pairs = read_minipairs(jsonl_path)
    if not pairs:
        raise RuntimeError(f"No valid pairs found in {jsonl_path}")

    correct = 0
    total = 0
    details: List[Dict] = []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (fp16 and device == "cuda")
        else contextlib.nullcontext()
    )

    with amp_ctx:
        for batch in tqdm(batch_iter(pairs, batch_size), total=(len(pairs) + batch_size - 1) // batch_size, desc="Evaluating"):
            # 将一批 (good, bad) 展平成 [good1, bad1, good2, bad2, ...]
            flat_texts: List[str] = []
            for g, b in batch:
                flat_texts.extend([g, b])

            losses = per_sample_loss(flat_texts, model, tokenizer, device)  # 长度=2*B
            # 还原为 (loss_good, loss_bad)
            for i, (g, b) in enumerate(batch):
                lg = losses[2 * i]
                lb = losses[2 * i + 1]
                pred = 1 if lg < lb else 0  # 1 表示更偏好 good
                label = 1
                is_correct = (pred == label)
                correct += int(is_correct)
                total += 1

                if save_details is not None:
                    details.append(
                        {
                            "sentence_good": g,
                            "sentence_bad": b,
                            "loss_good": lg,
                            "loss_bad": lb,
                            "loss_diff": lb - lg,
                            "correct": is_correct,
                        }
                    )

    accuracy = correct / total if total > 0 else float("nan")

    # 可选保存明细
    if save_details is not None:
        save_details.parent.mkdir(parents=True, exist_ok=True)
        with save_details.open("w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 清理
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single GPT-2 checkpoint on minipairs JSONL.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint dir (e.g., .../checkpoint-12345)")
    parser.add_argument("--jsonl", type=Path, required=True, help="JSONL file with fields: sentence_good, sentence_bad")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (pairs per forward pass)")
    parser.add_argument("--fp16", action="store_true", help="Enable autocast fp16 on CUDA")
    parser.add_argument("--device", type=str, default=None, help='Force device: "cuda" or "cpu" (default: auto)')
    parser.add_argument("--save-details", type=Path, default=None, help="Optional path to save per-example details as JSONL")
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
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {args.jsonl}")
    print(f"Total      : {res['total']}")
    print(f"Correct    : {res['correct']}")
    print(f"Accuracy   : {res['accuracy']:.4f}")


if __name__ == "__main__":
    # 禁用梯度
    torch.set_grad_enabled(False)
    main()
