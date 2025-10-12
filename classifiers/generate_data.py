#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified training data generator for animacy / NP-type / definiteness classifiers
===============================================================================

用法:
------
# 随机采样
python -m classifiers.generate_data --task animacy --max 2000 --strategy random

# 按类别均衡采样
python -m classifiers.generate_data --task nptype --max 4000 --strategy balanced
python -m classifiers.generate_data --task definiteness --max 4000 --strategy balanced
"""

import os
import json
import argparse
import csv
import random
import time
from collections import Counter
from tqdm import tqdm
from openai import OpenAI
from utils import DATA_PATH

client = OpenAI()

# ------------------ 分类配置 ------------------ #
TASK_CONFIGS = {
    "animacy": {
        "field": "animacy",
        "classes": ["animate", "inanimate"],
        "prompt": """Classify the animacy of the following noun phrase (NP).
Answer in a single word: "animate" or "inanimate".

Sentence: {sentence}
{np_type} NP: {np}

Definitions:
- "animate": refers to a living being such as a human or animal (e.g., the teacher, a cat, he, they)
- "inanimate": refers to a non-living entity, concept, or object (e.g., the book, the school, water, an idea)

Return exactly one of: animate, inanimate.
""",
    },
    "nptype": {
        "field": "nptype",
        "classes": ["pronoun", "common"],
        "prompt": """Classify the following noun phrase (NP) by its referential type.
Answer in a single word: "pronoun" or "common".

Sentence: {sentence}
{np_type} NP: {np}

Definitions:
- "pronoun": personal or demonstrative pronouns (e.g., I, you, he, she, it, they, this, that)
- "common": all other noun phrases, including proper names, definite and indefinite NPs (e.g., John, the teacher, a dog, some water)

Return exactly one of: pronoun, common.
""",
    },
    "definiteness": {
        "field": "definiteness",
        "classes": ["definite", "indef"],
        "prompt": """Classify the following noun phrase (NP) by its definiteness.
Answer in a single word: "definite" or "indef".

Sentence: {sentence}
{np_type} NP: {np}

Definitions:
- "definite": refers to an identifiable or unique referent known to both speaker and listener (e.g., the teacher, this book, John’s car)
- "indef": refers to a non-unique or non-specific referent (e.g., a student, some dog, any book, someone who could help)

Heuristics:
- If the NP uses "the", "this", "that", "my", "your", or other possessives, it is "definite".
- If it uses "a", "an", "some", "any", or appears in negation/quantified/hypothetical contexts (e.g., any, every, no, if, may, might), it is "indefinite".

Return exactly one of: definite, indef.
""",
    },
}


def call_openai_chat(sentence: str, np: str, np_type: str, task: str, model: str) -> str | None:
    """调用 OpenAI 模型对 NP 分类。"""
    cfg = TASK_CONFIGS[task]
    prompt = cfg["prompt"].format(sentence=sentence, np_type=np_type, np=np)

    for attempt in range(3):  # retry 机制
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
            if head in allowed:
                return head
            return None
        except Exception as e:
            print(f"[WARN] OpenAI call failed (attempt {attempt+1}/3): {e}")
            time.sleep(3)
    return None


def ensure_csv(path: str, field: str):
    """确保输出 CSV 存在表头。"""
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
    ensure_csv(out_csv, cfg["field"])

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

    random.shuffle(candidates)

    # ---------- 调用模型并采样 ---------- #
    cls_counter = Counter()
    with open(out_csv, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["sentence", "np", "np_role", cfg["field"]])
        pbar = tqdm(total=max_instances, desc=f"Generating {task} data")

        i = 0
        while sum(cls_counter.values()) < max_instances and i < len(candidates):
            sent, np_text, role = candidates[i]
            i += 1
            label = call_openai_chat(sent, np_text, role.capitalize(), task, model)
            if not label:
                continue

            if strategy == "balanced":
                per_class_limit = max_instances // len(cfg["classes"])
                if cls_counter[label] >= per_class_limit:
                    continue

            writer.writerow({"sentence": sent, "np": np_text, "np_role": role, cfg["field"]: label})
            fout.flush()
            cls_counter[label] += 1
            pbar.update(1)
            pbar.set_postfix({c: cls_counter.get(c, 0) for c in cfg["classes"]})

            if i > len(candidates) * 2:
                print("[WARN] Sampling stopped: exceeded 2× candidate size.")
                break

    print(f"[DONE] Wrote {sum(cls_counter.values())} examples to {out_csv}")
    print("[STATS]", dict(cls_counter))
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate training data for animacy, NP-type, or definiteness classifiers.")
    ap.add_argument("--task", choices=["animacy", "nptype", "definiteness"], required=True, help="Task type")
    ap.add_argument("--max", type=int, default=4000, help="Number of examples to generate")
    ap.add_argument("--strategy", choices=["random", "balanced"], default="random", help="Sampling strategy")
    ap.add_argument("--file", type=str, default="valid_verbs.jsonl", help="Input verbs file")
    ap.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    args = ap.parse_args()

    extract_training_data(
        task=args.task,
        max_instances=args.max,
        strategy=args.strategy,
        input_file=args.file,
        model=args.model,
    )
