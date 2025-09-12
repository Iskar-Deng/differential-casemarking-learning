#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perturbation for L1–L3 (LOCAL) with independent A/P rules.
==========================================================
该脚本用于对结构化语料进行局部扰动（local perturbation），
给 Agent/Patient 添加标记，用于生成 minimal pair 数据集。

【输入文件】
  - 默认读取 $DATA_PATH/structured 下的 train/valid/test *_verbs.jsonl

【输出文件】
  - 每个 split 会输出 3 个文件（affected/unaffected/invalid）
    共 9 个文件 + summary.json
  - 输出目录为 $DATA_PATH/perturbed_local/local_A{A_mode}-{A_markedness}_P{P_mode}-{P_markedness}

【参数说明】
  --strategy          目前仅支持 "local"
  --A_mode            Agent cutoff，可用值: none, human, animal, p12, p3, proper
                      多个 cutoff 用 '-' 拼接，例如 "animal-p12"
  --A_markedness      Agent 标记方向，可用值: forward, inverse, forward-inverse
                      多个用 '-' 对应 A_mode
  --P_mode / --P_markedness
                      Patient 的设置同上
  --input_dir         输入目录，默认 $DATA_PATH/structured
  --output_dir        输出目录，默认 $DATA_PATH/perturbed_local
  --debug             调试模式，最多处理 debug_limit 条有效样本
  --debug_limit       调试模式样本上限，默认 100

【断点续跑】
  - 如果中途因机器挂掉而停止，下次运行会跳过已处理行，继续处理剩余行。
  - 通过记录 "out_dir/{split}_progress.json" 保存进度。

【用法示例】
python -m perturbation.run_perturb \
  --strategy local \
  --A_mode none --A_markedness forward \
  --P_mode animal-p12 --P_markedness forward-inverse
"""

import os
import json
import argparse
import time
from glob import glob
from typing import Dict, List
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from utils import DATA_PATH, MODEL_PATH, AGENT_MARK, PATIENT_MARK

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANIMACY_HIER = ["human", "animal", "inanimate"]
NPDEF_HIER   = ["p12", "p3", "proper", "common"]
VALID_CUTOFFS = {"none", "human", "animal", "p12", "p3", "proper"}

def rank(label: str, hierarchy: List[str]) -> int:
    try:
        return hierarchy.index(label)
    except ValueError:
        return len(hierarchy)

def cutoff_hierarchy(cut: str) -> str | None:
    if cut in {"human", "animal"}:
        return "animacy"
    if cut in {"p12", "p3", "proper"}:
        return "npdef"
    if cut == "none":
        return None
    raise ValueError(f"Invalid cutoff: {cut}")

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

ANIMACY = _BertCls(
    os.path.join(MODEL_PATH, "animacy_bert_model"),
    id2lab={0: "human", 1: "animal", 2: "inanimate"}
)
NPDEF = _BertCls(
    os.path.join(MODEL_PATH, "npdef_bert_model"),
    id2lab={0: "p12", 1: "p3", 2: "proper", 3: "common"}
)

def predict_label(which: str, sentence: str, np_text: str) -> str:
    return ANIMACY.predict(sentence, np_text) if which == "animacy" else NPDEF.predict(sentence, np_text)

def parse_combo(arg: str) -> list[str]:
    return arg.split("-") if arg != "none" else []

def decide_mark(role: str, label: str, cutoff: str, markedness: str, hierarchy: List[str]) -> bool:
    lab_pos = rank(label, hierarchy)
    cut_pos = rank(cutoff, hierarchy)
    if role == "A":
        return lab_pos > cut_pos if markedness == "forward" else lab_pos <= cut_pos
    elif role == "P":
        return lab_pos <= cut_pos if markedness == "forward" else lab_pos > cut_pos
    raise ValueError(f"Unknown role: {role}")

def decide_mark_combo(role: str, label_dict: dict, cutoffs: list[str], marks: list[str]) -> bool:
    for cutoff, mark in zip(cutoffs, marks):
        hier = cutoff_hierarchy(cutoff)
        if hier is None:
            continue
        hierarchy = ANIMACY_HIER if hier == "animacy" else NPDEF_HIER
        lab = label_dict[hier]
        if not decide_mark(role, lab, cutoff, mark, hierarchy):
            return False
    return True

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

def process_one_jsonl(input_path, out_dir, strategy,
                      A_modes, A_marks, P_modes, P_marks,
                      debug, debug_limit):
    fname = os.path.basename(input_path)
    prefix = fname.replace("_verbs.jsonl", "")

    progress_file = os.path.join(out_dir, f"{prefix}_progress.json")
    processed_lines = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as pf:
            try:
                processed_lines = json.load(pf).get("processed", 0)
            except Exception:
                processed_lines = 0

    affected, unaffected, invalid = [], [], []
    stats = {"affected": 0, "unaffected": 0, "invalid": 0}
    processed_valid = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        total_lines = len(lines)
        lines = lines[processed_lines:]

        pbar = tqdm(total=total_lines, desc=f"{fname}", initial=processed_lines)
        last_update = time.time()

        for idx, line in enumerate(lines, processed_lines + 1):
            if debug and processed_valid >= debug_limit:
                break
            if not line.strip():
                continue

            data = json.loads(line)
            tokens = data.get("tokens") or []
            verbs  = data.get("verbs") or []
            sent_text = " ".join(t["text"] for t in tokens) if tokens else ""

            if not tokens or not verbs:
                invalid.append(sent_text); stats["invalid"] += 1
                continue

            has_valid, has_marked, spans = False, False, []
            for entry in verbs:
                if not is_valid_structure(entry): continue
                has_valid = True
                subj = entry["subject"]; obj = entry["objects"][0]
                if obj.get("dep") == "ccomp": continue

                label_dict = {}
                if any(cutoff_hierarchy(c) == "animacy" for c in A_modes+P_modes if c!="none"):
                    label_dict["animacy"] = ANIMACY.predict(sent_text, subj["text"])
                    label_dict["animacy_P"] = ANIMACY.predict(sent_text, obj["text"])
                if any(cutoff_hierarchy(c) == "npdef" for c in A_modes+P_modes if c!="none"):
                    label_dict["npdef"] = NPDEF.predict(sent_text, subj["text"])
                    label_dict["npdef_P"] = NPDEF.predict(sent_text, obj["text"])

                if A_modes:
                    subj_labels = {"animacy": label_dict.get("animacy"), "npdef": label_dict.get("npdef")}
                    if decide_mark_combo("A", subj_labels, A_modes, A_marks):
                        s = dict(subj); s["text"] = f"{s['text']} {AGENT_MARK}"
                        spans.append(s); has_marked = True

                if P_modes:
                    obj_labels = {"animacy": label_dict.get("animacy_P"), "npdef": label_dict.get("npdef_P")}
                    if decide_mark_combo("P", obj_labels, P_modes, P_marks):
                        o = dict(obj); o["text"] = f"{o['text']} {PATIENT_MARK}"
                        spans.append(o); has_marked = True

            if has_valid: processed_valid += 1
            if has_marked and spans:
                affected.append(apply_spans_to_tokens(tokens, spans)); stats["affected"] += 1
            elif has_valid:
                unaffected.append(sent_text); stats["unaffected"] += 1
            else:
                invalid.append(sent_text); stats["invalid"] += 1

            # 每分钟刷新一次 & 保存进度
            if time.time() - last_update >= 60 or idx == total_lines:
                pbar.n = idx
                pbar.set_postfix(stats)
                pbar.refresh()
                with open(progress_file, "w") as pf:
                    json.dump({"processed": idx}, pf)
                last_update = time.time()

    with open(os.path.join(out_dir, f"{prefix}_affected.txt"), "w") as f:
        for s in affected: f.write(s + "\n")
    with open(os.path.join(out_dir, f"{prefix}_unaffected.txt"), "w") as f:
        for s in unaffected: f.write(s + "\n")
    with open(os.path.join(out_dir, f"{prefix}_invalid.txt"), "w") as f:
        for s in invalid: f.write(s + "\n")

    return {"input": input_path, "stats": stats}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["local"], default="local")
    ap.add_argument("--A_mode", type=str, default="none")
    ap.add_argument("--A_markedness", type=str, default="forward")
    ap.add_argument("--P_mode", type=str, default="none")
    ap.add_argument("--P_markedness", type=str, default="forward")
    ap.add_argument("--input_dir", default=os.path.join(DATA_PATH, "structured"))
    ap.add_argument("--output_dir", default=os.path.join(DATA_PATH, "perturbed_local"))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_limit", type=int, default=100)
    args = ap.parse_args()

    A_modes = parse_combo(args.A_mode)
    P_modes = parse_combo(args.P_mode)
    A_marks = args.A_markedness.split("-") if args.A_markedness != "none" else []
    P_marks = args.P_markedness.split("-") if args.P_markedness != "none" else []

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
            ip, out_dir, args.strategy,
            A_modes, A_marks, P_modes, P_marks,
            args.debug, args.debug_limit
        ))

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Wrote outputs to {out_dir}")

if __name__ == "__main__":
    main()
