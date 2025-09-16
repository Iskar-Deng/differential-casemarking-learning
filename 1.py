#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check whether AGENT_MARK / PATIENT_MARK are single tokens in a trained checkpoint,
and print their token IDs + embedding norm.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

# === 配置路径 ===
CKPT = "checkpoints/local_Anone-forward_Pp3-forward_warmup/checkpoint-700"
AGENT_MARK = "🄰"
PATIENT_MARK = "🄿"

# === 加载 tokenizer & model ===
print(f"[INFO] Loading checkpoint from {CKPT}")
tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT)

print("\n[INFO] Special tokens map:", tokenizer.special_tokens_map)
print("[INFO] Additional special tokens:", tokenizer.additional_special_tokens)

def check_token(mark: str, name: str):
    toks = tokenizer.tokenize(mark)
    ids = tokenizer.convert_tokens_to_ids(toks)
    print(f"\n=== {name} ({mark}) ===")
    print(f"Tokenized pieces: {toks}  (count={len(toks)})")
    print(f"Token IDs: {ids}")
    if len(toks) == 1:
        emb = model.get_input_embeddings().weight[ids[0]].detach()
        print(f"Embedding norm: {emb.norm().item():.4f}")
    else:
        print("⚠️ 不是单一 token，会被拆成多个子词！")

# 检查两个标记
check_token(AGENT_MARK, "AGENT_MARK")
check_token(PATIENT_MARK, "PATIENT_MARK")
check_token("I", "I")
check_token("dog", "dog")

# 再测试一句话的分词情况
sample = f"the boy {PATIENT_MARK} saw the dog"
pieces = tokenizer.tokenize(sample)
print(f"\n[INFO] Sample sentence tokenization:\n{pieces}")
