#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perturbation for L1–L3 (LOCAL) with independent A/P rules.

- 输入：$DATA_PATH/structured 下的 train/valid/test *_verbs.jsonl
- 输出：每个 split 3 个文件（affected/unaffected/invalid），共 9 个文件 + summary.json
- A 与 P 可使用不同层级（animacy 或 npdef），由 cutoff 名推断
- cutoff 支持单一值 (L1/L2) 或组合 (L3)，例如：
    --P_mode animal-p12 --P_markedness forward-inverse
- 不允许使用 inanimate/common 作为 cutoff（因为已是零标基准）
- --debug: 每个 split 最多处理 100 条“有效样本”（有 tokens 且有 verbs）

用法示例：
python -m perturbation.run_perturb \
  --strategy local \
  --A_mode none --A_markedness forward \
  --P_mode animal-p12 --P_markedness forward-inverse \
  --debug
"""

import os
import json
import argparse
from glob import glob
from typing import Dict, List
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from utils import DATA_PATH, MODEL_PATH, AGENT_MARK, PATIENT_MARK

# ---------------- Perf ----------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Hierarchies ----------------
ANIMACY_HIER = ["human", "animal", "inanimate"]
NPDEF_HIER   = ["p12", "p3", "proper", "common"]

VALID_CUTOFFS = {"none", "human", "animal", "p12", "p3", "proper"}  # 不含 inanimate/common

def rank(label: str, hierarchy: List[str]) -> int:
    """位置越小越靠左（越突出）"""
    try:
        return hierarchy.index(label)
    except ValueError:
        return len(hierarchy)

def cutoff_hierarchy(cut: str) -> str | None:
    """根据 cutoff 名推断使用哪个层级"""
    if cut in {"human", "animal"}:
        return "animacy"
    if cut in {"p12", "p3", "proper"}:
        return "npdef"
    if cut == "none":
        return None
    raise ValueError(f"Invalid cutoff: {cut}")

# ---------------- Classifiers ----------------
class _BertCls:
    def __init__(self, model_dir: str, id2lab: Dict[int, str], max_length: int = 128):
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
    def predict(self, sentence: str, np_text: str) -> str:
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

def predict_label(which: str, sentence: str, np_text: str) -> str:
    if which == "animacy":
        return ANIMACY.predict(sentence, np_text)
    if which == "npdef":
        return NPDEF.predict(sentence, np_text)
    raise ValueError(f"Unknown predictor: {which}")

# ---------------- Rules ----------------
def parse_combo(arg: str) -> list[str]:
    """支持单值或组合，用 '-' 分隔"""
    return arg.split("-") if arg != "none" else []

def decide_mark(role: str, label: str, cutoff: str, markedness: str, hierarchy: List[str]) -> bool:
    """
    A（顺向）：在 cutoff 右边才标（严格 >）
    P（顺向）：在 cutoff 左边+等于才标（<=）
    逆向则取反
    """
    lab_pos = rank(label, hierarchy)
    cut_pos = rank(cutoff, hierarchy)
    if role == "A":
        if markedness == "forward":
            return lab_pos > cut_pos
        else:
            return lab_pos <= cut_pos
    elif role == "P":
        if markedness == "forward":
            return lab_pos <= cut_pos
        else:
            return lab_pos > cut_pos
    else:
        raise ValueError(f"Unknown role: {role}")

def decide_mark_combo(role: str, label_dict: dict, cutoffs: list[str], marks: list[str]) -> bool:
    """多维度组合判定：所有条件必须满足"""
    for cutoff, mark in zip(cutoffs, marks):
        hier = cutoff_hierarchy(cutoff)
        if hier is None:
            continue
        hierarchy = ANIMACY_HIER if hier == "animacy" else NPDEF_HIER
        lab = label_dict[hier]
        if not decide_mark(role, lab, cutoff, mark, hierarchy):
            return False
    return True

# ---------------- IO helpers ----------------
def is_valid_structure(entry: dict) -> bool:
    return bool(entry.get("subject")) and len(entry.get("objects", [])) == 1

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

def process_one_jsonl(
    input_path: str,
    out_dir: str,
    strategy: str,
    A_modes: list[str], A_marks: list[str],
    P_modes: list[str], P_marks: list[str],
    debug: bool, debug_limit: int
):
    fname = os.path.basename(input_path)
    prefix = fname.replace("_verbs.jsonl", "")
    out_affected   = os.path.join(out_dir, f"{prefix}_affected.txt")
    out_unaffected = os.path.join(out_dir, f"{prefix}_unaffected.txt")
    out_invalid    = os.path.join(out_dir, f"{prefix}_invalid.txt")

    affected, unaffected, invalid = [], [], []
    stats = {"affected": 0, "unaffected": 0, "invalid": 0}
    processed_valid = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        progress = tqdm(lines, desc=f"{fname}", total=len(lines))
        for line in progress:
            if debug and processed_valid >= debug_limit:
                break
            if not line.strip():
                continue

            data = json.loads(line)
            tokens = data.get("tokens") or []
            verbs  = data.get("verbs") or []
            sent_text = " ".join(t["text"] for t in tokens) if tokens else ""

            if not tokens or not verbs:
                invalid.append(sent_text)
                stats["invalid"] += 1
                continue

            has_valid, has_marked, spans = False, False, []
            for entry in verbs:
                if not is_valid_structure(entry):
                    continue
                has_valid = True
                subj = entry["subject"]
                obj  = entry["objects"][0]
                if obj.get("dep") == "ccomp":
                    continue

                # 先获取需要的 label
                label_dict = {}
                if any(cutoff_hierarchy(c) == "animacy" for c in A_modes+P_modes if c!="none"):
                    label_dict["animacy"] = ANIMACY.predict(sent_text, subj["text"])
                    label_dict["animacy_P"] = ANIMACY.predict(sent_text, obj["text"])
                if any(cutoff_hierarchy(c) == "npdef" for c in A_modes+P_modes if c!="none"):
                    label_dict["npdef"] = NPDEF.predict(sent_text, subj["text"])
                    label_dict["npdef_P"] = NPDEF.predict(sent_text, obj["text"])

                # Agent 规则
                if A_modes:
                    subj_labels = {"animacy": label_dict.get("animacy"),
                                   "npdef": label_dict.get("npdef")}
                    if decide_mark_combo("A", subj_labels, A_modes, A_marks):
                        s = dict(subj); s["text"] = f"{s['text']} {AGENT_MARK}"
                        spans.append(s); has_marked = True

                # Patient 规则
                if P_modes:
                    obj_labels = {"animacy": label_dict.get("animacy_P"),
                                  "npdef": label_dict.get("npdef_P")}
                    if decide_mark_combo("P", obj_labels, P_modes, P_marks):
                        o = dict(obj); o["text"] = f"{o['text']} {PATIENT_MARK}"
                        spans.append(o); has_marked = True

            if has_valid:
                processed_valid += 1

            if has_marked and spans:
                affected.append(apply_spans_to_tokens(tokens, spans))
                stats["affected"] += 1
            elif has_valid:
                unaffected.append(sent_text)
                stats["unaffected"] += 1
            else:
                invalid.append(sent_text)
                stats["invalid"] += 1

            # 更新进度条统计
            progress.set_postfix({
                "affected": stats["affected"],
                "unaff": stats["unaffected"],
                "invalid": stats["invalid"],
                "proc": processed_valid
            })

    os.makedirs(out_dir, exist_ok=True)
    with open(out_affected, "w", encoding="utf-8") as f:
        for s in affected: f.write(s + "\n")
    with open(out_unaffected, "w", encoding="utf-8") as f:
        for s in unaffected: f.write(s + "\n")
    with open(out_invalid, "w", encoding="utf-8") as f:
        for s in invalid: f.write(s + "\n")

    return {
        "input": input_path,
        "output": {"affected": out_affected, "unaffected": out_unaffected, "invalid": out_invalid},
        "stats": stats,
        "config": {
            "strategy": strategy,
            "A_modes": A_modes, "A_marks": A_marks,
            "P_modes": P_modes, "P_marks": P_marks,
            "debug": debug, "debug_limit": debug_limit
        }
    }

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["local"], default="local")
    ap.add_argument("--A_mode", type=str, default="none",
                    help="Cutoff(s) for Agent, e.g. 'animal' or 'animal-p12'")
    ap.add_argument("--A_markedness", type=str, default="forward",
                    help="Markedness for Agent, e.g. 'forward' or 'forward-inverse'")
    ap.add_argument("--P_mode", type=str, default="none",
                    help="Cutoff(s) for Patient, e.g. 'human' or 'animal-p12'")
    ap.add_argument("--P_markedness", type=str, default="forward",
                    help="Markedness for Patient, e.g. 'forward' or 'forward-inverse'")
    ap.add_argument("--input_dir", default=os.path.join(DATA_PATH, "structured"))
    ap.add_argument("--output_dir", default=os.path.join(DATA_PATH, "perturbed_local"))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_limit", type=int, default=100)
    args = ap.parse_args()

    # 拆分组合
    A_modes = parse_combo(args.A_mode)
    P_modes = parse_combo(args.P_mode)
    A_marks = args.A_markedness.split("-") if args.A_markedness != "none" else []
    P_marks = args.P_markedness.split("-") if args.P_markedness != "none" else []

    if len(A_modes) != len(A_marks) and A_modes:
        raise ValueError(f"A_mode/A_markedness length mismatch: {A_modes} vs {A_marks}")
    if len(P_modes) != len(P_marks) and P_modes:
        raise ValueError(f"P_mode/A_markedness length mismatch: {P_modes} vs {P_marks}")

    inputs = sorted(glob(os.path.join(args.input_dir, "*_verbs.jsonl")))
    if not inputs:
        raise FileNotFoundError(f"No *_verbs.jsonl under {args.input_dir}")

    out_dir = os.path.join(
        args.output_dir,
        f"local_A{args.A_mode}-{args.A_markedness}_P{args.P_mode}-{args.P_markedness}"
    )
    os.makedirs(out_dir, exist_ok=True)

    summaries = []
    for ip in inputs:
        summaries.append(process_one_jsonl(
            ip, out_dir,
            args.strategy,
            A_modes, A_marks,
            P_modes, P_marks,
            debug=args.debug, debug_limit=args.debug_limit
        ))

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote outputs to: {out_dir}")
    for s in summaries:
        print(" -", s["output"]["affected"])
        print(" -", s["output"]["unaffected"])
        print(" -", s["output"]["invalid"])
    print(" -", summary_path)

if __name__ == "__main__":
    main()
