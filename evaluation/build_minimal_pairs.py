#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build minimal pairs from structured test set (L1–L3 local rules, P-only when A_mode=none).

- 输入：$DATA_PATH/structured/test_verbs.jsonl
- 输出：$EVALUATION_PATH/casemarking/local_Amode-mark_Pmode-mark/test_minimal_pairs.jsonl
- 规则：
  * 仅在所启用的角色上构造最小对；若 A_mode=none，则严格只考虑 P（Patient）。
  * 一半 pair：good=按规则应当标记 → 在同一 P 处加标；bad=同一处不标。
  * 另一半 pair：good=按规则不应标记 → 不标；bad=同一 P 处强制加标（“给不该加的P加上标记”）。
  * 保证 minimal：两句仅在一个 NP 是否带标记上有差异，其他文本一致。
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

# ---------------- Helpers ----------------
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
    if not spans:
        return " ".join(t["text"] for t in tokens)
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

def should_mark_role(role, subj_labels, obj_labels, A_modes, A_marks, P_modes, P_marks):
    if role == "A":
        if not A_modes:  # 未启用 A，绝不标 A
            return None  # 用 None 表示“不考虑该角色”
        return decide_mark_combo("A", subj_labels, A_modes, A_marks)
    else:
        if not P_modes:
            return None
        return decide_mark_combo("P", obj_labels, P_modes, P_marks)

def build_pair_on_role(tokens, subj, obj, role, good_is_marked):
    base_tokens = [dict(t) for t in tokens]
    span = dict(subj if role == "A" else obj)
    span["text"] = f"{span['text']} " + (AGENT_MARK if role == "A" else PATIENT_MARK)
    marked_sent = apply_spans_to_tokens(base_tokens, [span])
    unmarked_sent = " ".join(t["text"] for t in base_tokens)
    if good_is_marked:
        return marked_sent, unmarked_sent
    else:
        return unmarked_sent, marked_sent

def count_marks(s):
    return s.count(AGENT_MARK) + s.count(PATIENT_MARK)

def stripped(s):
    return s.replace(AGENT_MARK, "").replace(PATIENT_MARK, "").replace("  ", " ").strip()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A_mode", type=str, default="none")
    ap.add_argument("--A_markedness", type=str, default="forward")
    ap.add_argument("--P_mode", type=str, default="none")
    ap.add_argument("--P_markedness", type=str, default="forward")
    ap.add_argument("--num_pairs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    random.seed(args.seed)

    A_modes = parse_combo(args.A_mode)
    P_modes = parse_combo(args.P_mode)
    A_marks = args.A_markedness.split("-") if args.A_markedness != "none" else []
    P_marks = args.P_markedness.split("-") if args.P_markedness != "none" else []

    # 若 A_mode=none，则不考虑 A；仅在 P 上构造最小对
    roles_to_consider = []
    if A_modes:
        roles_to_consider.append("A")
    if P_modes:
        roles_to_consider.append("P")
    if not roles_to_consider:
        raise ValueError("At least one of A_mode or P_mode must be enabled to build pairs.")

    input_path = os.path.join(DATA_PATH, "structured", "test_verbs.jsonl")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No test_verbs.jsonl under {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)

    out_dir = Path(EVALUATION_PATH) / f"casemarking/local_A{args.A_mode}-{args.A_markedness}_P{args.P_mode}-{args.P_markedness}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "test_minimal_pairs.jsonl"

    half = args.num_pairs // 2
    good_marked_target = 0
    good_unmarked_target = 0
    results = []

    for line in tqdm(lines, desc="Building minimal pairs", total=len(lines)):
        if len(results) >= args.num_pairs:
            break
        data = json.loads(line)
        tokens = data.get("tokens") or []
        verbs = data.get("verbs") or []
        if not tokens or not verbs:
            continue

        sent_text = " ".join(t["text"] for t in tokens)

        # 是否需要调用分类器：只在启用对应层级的情况下预测
        need_anim = any(cutoff_hierarchy(c) == "animacy" for c in A_modes + P_modes if c != "none")
        need_npdef = any(cutoff_hierarchy(c) == "npdef" for c in A_modes + P_modes if c != "none")

        for entry in verbs:
            subj = entry.get("subject")
            objs = entry.get("objects") or []
            if not subj or len(objs) != 1:
                continue
            obj = objs[0]
            if obj.get("dep") == "ccomp":
                continue

            subj_labels = {
                "animacy": ANIMACY.predict(sent_text, subj["text"]) if need_anim else None,
                "npdef":   NPDEF.predict(sent_text, subj["text"])   if need_npdef else None,
            }
            obj_labels = {
                "animacy": ANIMACY.predict(sent_text, obj["text"]) if need_anim else None,
                "npdef":   NPDEF.predict(sent_text, obj["text"])   if need_npdef else None,
            }

            # 只遍历允许的角色（例如 A_mode=none 时，这里只会有 'P'）
            for role in roles_to_consider:
                sm = should_mark_role(role, subj_labels, obj_labels, A_modes, A_marks, P_modes, P_marks)
                if sm is None:
                    continue  # 该角色未启用，跳过（保险）

                # 前一半 pair：选择“应标记”的样本 → good 带标
                # 后一半 pair：选择“应不标记”的样本 → good 不带标，bad 强制加标
                pick_mark = (good_marked_target < half)
                if pick_mark and not sm:
                    continue
                if (not pick_mark) and sm:
                    continue

                good, bad = build_pair_on_role(tokens, subj, obj, role, good_is_marked=pick_mark)

                # minimal & 一致性检查
                if abs(count_marks(good) - count_marks(bad)) != 1:
                    continue
                if stripped(good) != stripped(bad):
                    continue

                results.append({"sentence_good": good, "sentence_bad": bad})
                if pick_mark:
                    good_marked_target += 1
                else:
                    good_unmarked_target += 1
                break  # 选中一个 entry+role 即可，保证只改一个位置
            if len(results) >= args.num_pairs:
                break

    # 若 num_pairs 为奇数，可能少 1；此处按目前逻辑保持不变

    with open(out_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[DONE] 生成 minimal pairs: {len(results)} 条")
    print(f"[INFO] good(有标记)={good_marked_target} good(无标记)={good_unmarked_target}")
    print(f"[INFO] 输出文件: {out_file}")

if __name__ == "__main__":
    main()
