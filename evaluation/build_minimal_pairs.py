#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build minimal pairs from structured test set (L1–L3 local rules).

- 输入：$DATA_PATH/structured/test_verbs.jsonl
- 输出：$EVALUATION_PATH/casemarking/local_Amode-mark_Pmode-mark/test_minimal_pairs.jsonl
- good/bad 双向配对：一半有标记作 good，一半无标记作 good
- bad 强制反向标记（即顺向改成逆向）
- 随机采样直到满足 num_pairs
"""

import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from utils import DATA_PATH, EVALUATION_PATH, MODEL_PATH, AGENT_MARK, PATIENT_MARK

# ---------------- Perf ----------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Hierarchies ----------------
ANIMACY_HIER = ["human", "animal", "inanimate"]
NPDEF_HIER   = ["p12", "p3", "proper", "common"]

def rank(label, hierarchy):
    try:
        return hierarchy.index(label)
    except ValueError:
        return len(hierarchy)

def cutoff_hierarchy(cut):
    if cut in {"human", "animal"}:
        return "animacy"
    if cut in {"p12", "p3", "proper"}:
        return "npdef"
    if cut == "none":
        return None
    raise ValueError(f"Invalid cutoff: {cut}")

# ---------------- Classifiers ----------------
class _BertCls:
    def __init__(self, model_dir, id2lab, max_length=128):
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
        inputs = self.tok(text, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=self.maxlen)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        lab = int(torch.argmax(logits, dim=1).item())
        return self.id2lab[lab]

ANIMACY = _BertCls(
    os.path.join(MODEL_PATH, "animacy_bert_model"),
    id2lab={0: "human", 1: "animal", 2: "inanimate"}
)
NPDEF = _BertCls(
    os.path.join(MODEL_PATH, "npdef_bert_model"),
    id2lab={0: "p12", 1: "p3", 2: "proper", 3: "common"}
)

def parse_combo(arg):
    return arg.split("-") if arg != "none" else []

def decide_mark(role, label, cutoff, markedness, hierarchy):
    lab_pos = rank(label, hierarchy)
    cut_pos = rank(cutoff, hierarchy)
    if role == "A":
        return lab_pos > cut_pos if markedness == "forward" else lab_pos <= cut_pos
    else:
        return lab_pos <= cut_pos if markedness == "forward" else lab_pos > cut_pos

def decide_mark_combo(role, label_dict, cutoffs, marks):
    for cutoff, mark in zip(cutoffs, marks):
        hier = cutoff_hierarchy(cutoff)
        if hier is None:
            continue
        hierarchy = ANIMACY_HIER if hier == "animacy" else NPDEF_HIER
        lab = label_dict[hier]
        if lab is None:
            return False
        if not decide_mark(role, lab, cutoff, mark, hierarchy):
            return False
    return True

def apply_spans_to_tokens(tokens, spans):
    spans = sorted(spans, key=lambda s: s["span"][0])
    out, i = [], 0
    for sp in spans:
        s, e = sp["span"]
        while i < s and i < len(tokens):
            out.append(tokens[i]["text"]); i += 1
        out.append(sp["text"])
        i = max(i, e + 1)
    while i < len(tokens):
        out.append(tokens[i]["text"]); i += 1
    return " ".join(out)

def mark_sentence(data, A_modes, A_marks, P_modes, P_marks, invert=False):
    tokens = data.get("tokens") or []
    verbs = data.get("verbs") or []
    sent_text = " ".join(t["text"] for t in tokens)
    spans, has_marked = [], False

    for entry in verbs:
        subj = entry.get("subject")
        objects = entry.get("objects") or []
        if not subj or len(objects) != 1:
            continue
        obj = objects[0]
        if obj.get("dep") == "ccomp":
            continue

        label_dict = {}
        if any(cutoff_hierarchy(c) == "animacy" for c in A_modes+P_modes if c!="none"):
            label_dict["animacy"] = ANIMACY.predict(sent_text, subj["text"])
            label_dict["animacy_P"] = ANIMACY.predict(sent_text, obj["text"])
        if any(cutoff_hierarchy(c) == "npdef" for c in A_modes+P_modes if c!="none"):
            label_dict["npdef"] = NPDEF.predict(sent_text, subj["text"])
            label_dict["npdef_P"] = NPDEF.predict(sent_text, obj["text"])

        # Agent
        if A_modes:
            subj_labels = {"animacy": label_dict.get("animacy"),
                           "npdef": label_dict.get("npdef")}
            A_marks_eff = [
                "inverse" if invert and m == "forward" else
                "forward" if invert and m == "inverse" else m
                for m in A_marks
            ]
            if decide_mark_combo("A", subj_labels, A_modes, A_marks_eff):
                s = dict(subj); s["text"] = f"{s['text']} {AGENT_MARK}"
                spans.append(s); has_marked = True

        # Patient
        if P_modes:
            obj_labels = {"animacy": label_dict.get("animacy_P"),
                          "npdef": label_dict.get("npdef_P")}
            P_marks_eff = [
                "inverse" if invert and m == "forward" else
                "forward" if invert and m == "inverse" else m
                for m in P_marks
            ]
            if decide_mark_combo("P", obj_labels, P_modes, P_marks_eff):
                o = dict(obj); o["text"] = f"{o['text']} {PATIENT_MARK}"
                spans.append(o); has_marked = True

    return apply_spans_to_tokens(tokens, spans) if has_marked else sent_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A_mode", type=str, default="none")
    ap.add_argument("--A_markedness", type=str, default="forward")
    ap.add_argument("--P_mode", type=str, default="none")
    ap.add_argument("--P_markedness", type=str, default="forward")
    ap.add_argument("--num_pairs", type=int, required=True)
    args = ap.parse_args()

    A_modes = parse_combo(args.A_mode)
    P_modes = parse_combo(args.P_mode)
    A_marks = args.A_markedness.split("-") if args.A_markedness != "none" else []
    P_marks = args.P_markedness.split("-") if args.P_markedness != "none" else []

    input_path = os.path.join(DATA_PATH, "structured", "test_verbs.jsonl")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No test_verbs.jsonl under {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)

    out_dir = Path(EVALUATION_PATH) / f"casemarking/local_A{args.A_mode}-{args.A_markedness}_P{args.P_mode}-{args.P_markedness}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "test_minimal_pairs.jsonl"

    good_count, bad_count = 0, 0
    results = []

    for line in tqdm(lines, desc="Building minimal pairs", total=len(lines)):
        if good_count >= args.num_pairs // 2 and bad_count >= args.num_pairs // 2:
            break
        data = json.loads(line)
        if not data.get("tokens") or not data.get("verbs"):
            continue

        marked = mark_sentence(data, A_modes, A_marks, P_modes, P_marks, invert=False)
        unmarked = mark_sentence(data, A_modes, A_marks, P_modes, P_marks, invert=True)
        if marked == unmarked:
            continue  # 无法形成最小对

        if good_count < args.num_pairs // 2:
            results.append({"sentence_good": marked, "sentence_bad": unmarked})
            good_count += 1
        elif bad_count < args.num_pairs // 2:
            results.append({"sentence_good": unmarked, "sentence_bad": marked})
            bad_count += 1

    with open(out_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[DONE] 生成 minimal pairs: {len(results)} 条")
    print(f"[INFO] good(有标记)={good_count} bad(无标记)={bad_count}")
    print(f"[INFO] 输出文件: {out_file}")

if __name__ == "__main__":
    main()
