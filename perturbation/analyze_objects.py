#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze object (Patient) distribution across animacy and NP-def classifiers.

- 输入: $DATA_PATH/structured 下的 *_verbs.jsonl
- 输出: 一个 CSV 表格，统计宾语在 animacy × npdef 的组合分布
- 控制台同时打印各 split 的分布 + 总体分布

用法:
python -m perturbation.analyze_objects
"""

import os
import json
import argparse
from glob import glob
from collections import Counter

import pandas as pd
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from utils import DATA_PATH, MODEL_PATH

# ---------------- Perf ----------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Classifier wrapper ----------------
class _BertCls:
    def __init__(self, model_dir, id2lab, max_length=128):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model dir not found: {model_dir}")
        self.tok = BertTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
        self.model.eval()
        if device.type == "cuda":
            self.model.half()
        self.id2lab = id2lab
        self.maxlen = max_length

    @torch.inference_mode()
    def predict(self, sentence, np_text):
        text = f"{sentence} [NP] {np_text}"
        inputs = self.tok(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.maxlen
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        lab = int(torch.argmax(logits, dim=1).item())
        return self.id2lab[lab]

# 初始化两个分类器
ANIMACY = _BertCls(
    os.path.join(MODEL_PATH, "animacy_bert_model"),
    id2lab={0: "human", 1: "animal", 2: "inanimate"}
)
NPDEF = _BertCls(
    os.path.join(MODEL_PATH, "npdef_bert_model"),
    id2lab={0: "p12", 1: "p3", 2: "proper", 3: "common"}
)

# ---------------- Helpers ----------------
def is_valid_structure(entry):
    return bool(entry.get("subject")) and len(entry.get("objects", [])) == 1

def analyze_file(path):
    counter = Counter()
    with open(path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=os.path.basename(path)):
            if not line.strip():
                continue
            data = json.loads(line)
            tokens = data.get("tokens") or []
            verbs = data.get("verbs") or []
            if not tokens or not verbs:
                continue
            sent_text = " ".join(t["text"] for t in tokens)
            for entry in verbs:
                if not is_valid_structure(entry):
                    continue
                obj = entry["objects"][0]
                if obj.get("dep") == "ccomp":
                    continue
                animacy_label = ANIMACY.predict(sent_text, obj["text"])
                npdef_label = NPDEF.predict(sent_text, obj["text"])
                combo = f"{animacy_label}-{npdef_label}"
                counter[combo] += 1
    return counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=os.path.join(DATA_PATH, "structured"))
    ap.add_argument("--output_dir", default=os.path.join(DATA_PATH, "stats"))
    args = ap.parse_args()

    inputs = sorted(glob(os.path.join(args.input_dir, "*_verbs.jsonl")))
    if not inputs:
        raise FileNotFoundError(f"No *_verbs.jsonl under {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_counters = {}
    total_counter = Counter()
    for ip in inputs:
        c = analyze_file(ip)
        split = os.path.basename(ip).replace("_verbs.jsonl", "")
        all_counters[split] = c
        total_counter.update(c)

    # 转成 DataFrame
    df = pd.DataFrame.from_dict(all_counters, orient="index").fillna(0).astype(int)
    df.loc["TOTAL"] = df.sum(axis=0)

    # 保存
    out_path = os.path.join(args.output_dir, "object_distribution.csv")
    df.to_csv(out_path)
    print(f"\n[RESULT] Saved object distribution table to {out_path}\n")
    print(df)

if __name__ == "__main__":
    main()
