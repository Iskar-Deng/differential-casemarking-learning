#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a single GPT-2 checkpoint on a minipairs JSONL dataset.

Usage:
    python -m evaluation.eval_single_ckpt \
        --checkpoint /workspace/differential-casemarking-learning/checkpoints/independent_Anone_Panimate/checkpoint-70000 \
        --jsonl evaluation/minimal_pairs/independent_Anone_Panimate/valid_test_minimal_pairs.jsonl \
        --save-details results/independent_Anone_Panimate.jsonl
    python -m evaluation.eval_single_ckpt \
        --checkpoint /workspace/differential-casemarking-learning/checkpoints/independent_Anone_Pdefinite/checkpoint-70000 \
        --jsonl evaluation/minimal_pairs/independent_Anone_Pdefinite/valid_test_minimal_pairs.jsonl \
        --save-details results/independent_Anone_Pdefinite.jsonl
    python -m evaluation.eval_single_ckpt \
        --checkpoint /workspace/differential-casemarking-learning/checkpoints/independent_Anone_Pdefinite_inv/checkpoint-70000 \
        --jsonl evaluation/minimal_pairs/independent_Anone_Pdefinite_inv/valid_test_minimal_pairs.jsonl \
        --save-details results/independent_Anone_Pdefinite_inv.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import contextlib

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm.auto import tqdm


# =====================================================
# Helpers
# =====================================================
def load_tokenizer(checkpoint: Path) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint: Path, device: str) -> GPT2LMHeadModel:
    """
    自动检测并修复 torch.compile() 导致的 '_orig_mod.' 前缀问题，
    同时支持 Hugging Face safetensors / 分片权重。
    """
    import torch
    from safetensors.torch import load_file as safe_load

    bin_path = Path(checkpoint) / "pytorch_model.bin"
    safe_path = Path(checkpoint) / "model.safetensors"
    index_path = Path(checkpoint) / "pytorch_model.bin.index.json"

    # ---- 判断权重文件格式 ----
    if safe_path.exists():
        print(f"[Info] 检测到 safetensors 权重文件: {safe_path.name}")
        state_dict = safe_load(str(safe_path))
    elif bin_path.exists():
        print(f"[Info] 检测到 pytorch_model.bin 权重文件")
        state_dict = torch.load(bin_path, map_location="cpu")
    elif index_path.exists():
        print(f"[Info] 检测到分片权重 index 文件: {index_path.name}")
        state_dict = None  # 交给 from_pretrained 自动解析
    else:
        raise FileNotFoundError(f"未找到可用的权重文件 (safetensors/bin/index) 于 {checkpoint}")

    # ---- 检查并修正 '_orig_mod.' 前缀 ----
    has_orig_mod = state_dict is not None and any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if has_orig_mod:
        print(f"[Fix] 检测到 torch.compile() 权重前缀 '_orig_mod.' → 正在修正...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # ✨ 关键改动：手动初始化 GPT2LMHeadModel，然后加载修正后的权重
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config.from_pretrained(checkpoint)
        model = GPT2LMHeadModel(config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[Warn] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(str(checkpoint))


    model.to(device)
    model.eval()
    return model




def read_minipairs(jsonl_path: Path) -> List[Tuple[str, str]]:
    """读取 sentence_good / sentence_bad 对"""
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
    """计算每个样本的平均 token-level loss"""
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


# =====================================================
# Evaluation
# =====================================================
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
    details = []

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
                pred_correct = lg < lb
                correct += int(pred_correct)
                total += 1
                (delta_correct if pred_correct else delta_wrong).append(diff)

                if save_details is not None:
                    details.append({
                        "sentence_good": g,
                        "sentence_bad": b,
                        "loss_good": lg,
                        "loss_bad": lb,
                        "loss_diff": diff,
                        "correct": pred_correct,
                    })

    accuracy = correct / total if total > 0 else float("nan")
    mean_delta = sum(delta_losses) / len(delta_losses) if delta_losses else float("nan")
    mean_delta_correct = sum(delta_correct) / len(delta_correct) if delta_correct else float("nan")
    mean_delta_wrong = sum(delta_wrong) / len(delta_wrong) if delta_wrong else float("nan")

    if save_details is not None:
        save_details.parent.mkdir(parents=True, exist_ok=True)
        with save_details.open("w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "checkpoint": str(checkpoint),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mean_delta": mean_delta,
        "mean_delta_correct": mean_delta_correct,
        "mean_delta_wrong": mean_delta_wrong,
    }


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate a single GPT-2 checkpoint on a minipairs JSONL dataset.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint folder")
    parser.add_argument("--jsonl", type=Path, required=True, help="JSONL file with sentence_good / sentence_bad pairs")
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


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
