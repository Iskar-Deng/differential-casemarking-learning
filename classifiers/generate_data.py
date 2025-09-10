#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified training data generator for animacy and NP-def classifiers.

用法:
------
# 随机采样
python -m classifiers.generate_data --task animacy --max 5000 --strategy random

# 按类别均衡采样
python -m classifiers.generate_data --task npdef --max 4000 --strategy balanced
"""

import os
import json
import argparse
import csv
import random
from collections import Counter
from tqdm import tqdm
from openai import OpenAI
from utils import DATA_PATH

client = OpenAI()

# ------------------ 分类配置 ------------------ #
TASK_CONFIGS = {
    "animacy": {
        "field": "animacy",
        "classes": ["human", "animal", "inanimate"],
        "prompt": """Classify the animacy of the following NP. Answer with one of the categories: "human", "animal", or "inanimate". Answer in a single word without quotes.

Sentence: {sentence}
{np_type} NP: {np}

Respond with one of the following categories:
- "human": refers to a person or people (e.g., the teacher, him, John)
- "animal": refers to an animal or animals (e.g., the dog, a bird)
- "inanimate": refers to a non-living thing, place, object or activity (e.g., the book, the school, a rock, running)
""",
    },
    "npdef": {
        "field": "np_person",
        "classes": ["p12", "p3", "proper", "common"],
        "prompt": """Classify the following NP into one of four referential classes. 
Answer in a single word (no quotes): "p12", "p3", "proper", or "common".

Sentence: {sentence}
{np_type} NP: {np}

Use these definitions:
- "p12": first- or second-person pronouns (e.g., I, me, we, us, you, myself, ourselves, your)
- "p3": third-person pronouns only (e.g., he, him, she, her, it, they, them, himself, herself, itself, themselves, their)
- "proper": proper names or unique proper nouns (e.g., John, Mary, Paris, Microsoft)
- "common": all other common noun phrases (definite/indefinite NPs like "the teacher", "a dog", "some water", "this book")

Return exactly one of: p12, p3, proper, common.
""",
    },
}


def call_openai_chat(sentence: str, np: str, np_type: str, task: str, model: str) -> str | None:
    cfg = TASK_CONFIGS[task]
    prompt = cfg["prompt"].format(sentence=sentence, np_type=np_type, np=np)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2,
        )
        result = (resp.choices[0].message.content or "").strip().lower().replace('"', "")
        allowed = set(cfg["classes"])
        if result in allowed:
            return result
        head = (result.split() or [""])[0]
        return head if head in allowed else None
    except Exception as e:
        print(f"[ERROR] OpenAI call failed: {e}")
        return None


def ensure_csv(path: str, field: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=["sentence", "np", "np_role", field])
            writer.writeheader()


def extract_training_data(task: str, max_instances: int, strategy: str, input_file: str, model: str):
    cfg = TASK_CONFIGS[task]
    structured_dir = os.path.join(DATA_PATH, "structured")
    if not os.path.isfile(input_file):
        input_file = os.path.join(structured_dir, input_file)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found")

    stem = os.path.splitext(os.path.basename(input_file))[0]
    out_csv = os.path.join(DATA_PATH, f"training_data_{task}_{stem.replace('_verbs','')}.csv")
    ensure_csv(out_csv, cfg["field"])   # ← 修复点

    # ---------- 收集候选 NP ---------- #
    candidates = []
    with open(input_file, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            tokens = data.get("tokens", [])
            verbs = data.get("verbs") or []
            if not tokens or not verbs:
                continue
            sentence = " ".join(tok.get("text", "") for tok in tokens)
            for entry in verbs:
                subj = entry.get("subject")
                objs = entry.get("objects") or []
                if not subj or len(objs) != 1:
                    continue
                if objs[0].get("dep") == "ccomp":
                    continue
                candidates.append((sentence, subj["text"], "subject"))
                candidates.append((sentence, objs[0]["text"], "object"))

    print(f"[INFO] Collected {len(candidates):,} candidate NPs")

    # 打乱顺序，实现“全局随机采样”
    random.shuffle(candidates)

    # ---------- 采样并调用模型 ---------- #
    cls_counter = Counter()
    with open(out_csv, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["sentence", "np", "np_role", cfg["field"]])
        pbar = tqdm(total=max_instances, desc=f"Generating {task} data")

        i = 0
        while sum(cls_counter.values()) < max_instances and i < len(candidates):
            sent, np, role = candidates[i]
            i += 1
            label = call_openai_chat(sent, np, role.capitalize(), task, model)
            if not label:
                continue

            if strategy == "balanced":
                # 每类上限
                if cls_counter[label] >= max_instances // len(cfg["classes"]):
                    continue

            writer.writerow({"sentence": sent, "np": np, "np_role": role, cfg["field"]: label})
            fout.flush()
            cls_counter[label] += 1
            pbar.update(1)
            # 在进度条上动态显示类别计数
            pbar.set_postfix({c: cls_counter.get(c, 0) for c in cfg["classes"]})

    print(f"[DONE] Wrote {sum(cls_counter.values())} examples to {out_csv}")
    print("[STATS]", dict(cls_counter))
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate training data for animacy or NP-def classifiers.")
    ap.add_argument("--task", choices=["animacy", "npdef"], required=True, help="Task type")
    ap.add_argument("--max", type=int, default=4000, help="Number of examples to generate")
    ap.add_argument("--strategy", choices=["random", "balanced"], default="random", help="Sampling strategy")
    ap.add_argument("--file", type=str, default="valid_verbs.jsonl", help="Input verbs file")
    ap.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model")
    args = ap.parse_args()

    extract_training_data(
        task=args.task,
        max_instances=args.max,
        strategy=args.strategy,
        input_file=args.file,
        model=args.model,
    )
