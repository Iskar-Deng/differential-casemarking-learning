#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_invalid_txts.py
========================
一次性提取 structured_labeled/*.invalid.jsonl 文件中的文本内容，
保存为可复用的 TXT 文件。

输出:
  data/invalid_texts/
    ├── train_invalid.txt
    ├── valid_invalid.txt
    ├── test_invalid.txt

之后各系统的 run_perturb_v2.py 就无需再读取 44 GB 的 invalid 源文件。
"""

import os
import json
from tqdm import tqdm
from utils import DATA_PATH

# ------------------ 路径设置 ------------------
SRC_DIR = os.path.join(DATA_PATH, "structured_labeled")
OUT_DIR = os.path.join(DATA_PATH, "invalid_texts")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ 目标文件名映射 ------------------
FILES = {
    "train": "train_verbs_labeled.invalid.jsonl",
    "valid": "valid_verbs_labeled.invalid.jsonl",
    "test": "test_verbs_labeled.invalid.jsonl",
}

# ------------------ 主逻辑 ------------------
for split, fname in FILES.items():
    in_path = os.path.join(SRC_DIR, fname)
    out_path = os.path.join(OUT_DIR, f"{split}_invalid.txt")

    if not os.path.exists(in_path):
        print(f"[WARN] Missing {in_path}, skipped.")
        continue

    print(f"[INFO] Processing {in_path}")
    pbar = tqdm(desc=f"{split}_invalid", unit="lines")

    count = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            tokens = data.get("tokens") or []
            if not tokens:
                continue
            sent = " ".join(tok["text"] for tok in tokens)
            fout.write(sent + "\n")
            count += 1
            if count % 10000 == 0:
                pbar.update(10000)
        pbar.update(count % 10000)
    pbar.close()
    print(f"[DONE] Wrote {count} sentences → {out_path}")
