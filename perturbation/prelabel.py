#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prelabel NP-level features (animacy / nptype / definiteness) and split into two sets.
=====================================================================================

本脚本只做“中间量预计算”，不做任何扰动。
对 $DATA_PATH/structured 下的 *_verbs.jsonl 做如下处理：
  1) 若句子中存在至少一个满足 is_valid_structure(entry) 的 verb entry → 写入 *_verbs_labeled.ok.jsonl
     并为该句子里的 subject / object NP 写入：
        - animacy      ∈ {animate, inanimate}
        - nptype       ∈ {pronoun, common}
        - definiteness ∈ {definite, indef}
  2) 其它（无 verbs 或无有效结构） → 写入 *_verbs_labeled.invalid.jsonl（原样或仅带基础字段）

支持断点续跑（<split>_progress.json），可随时中断/续跑。
"""

import os
import json
import argparse
import time
from glob import glob
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from utils import DATA_PATH, MODEL_PATH

# -------------------- Torch setup --------------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Helper --------------------
def is_valid_structure(entry: dict) -> bool:
    """与后续扰动判定保持一致：subject 存在 + objects 恰好 1 + object 不是 ccomp。"""
    if not entry.get("subject"):
        return False
    objs = entry.get("objects") or []
    if len(objs) != 1:
        return False
    if objs[0].get("dep") == "ccomp":
        return False
    return True


# -------------------- Classifier wrapper --------------------
class BertClassifier:
    def __init__(self, model_dir: str, id2lab: Dict[int, str], max_length: int = 128):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"[ERR] Model dir not found: {model_dir}")
        self.tokenizer = BertTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model: nn.Module = BertForSequenceClassification.from_pretrained(model_dir).to(device)
        self.model.eval()
        if device.type == "cuda":
            self.model.half()
        self.id2lab = id2lab
        self.maxlen = max_length

    @torch.inference_mode()
    def predict_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 64) -> List[str]:
        """pairs: [(sentence, np_text), ...] -> labels"""
        out_labels: List[str] = []
        if not pairs:
            return out_labels
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            texts = [f"{s} [NP] {np}" for (s, np) in chunk]
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.maxlen
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            ids = torch.argmax(logits, dim=1).tolist()
            out_labels.extend([self.id2lab[j] for j in ids])
        return out_labels


def load_classifiers(
    use_animacy: bool = True,
    use_nptype: bool = True,
    use_definiteness: bool = True,
    max_length: int = 128,
) -> Dict[str, Optional[BertClassifier]]:
    clsf = {"animacy": None, "nptype": None, "definiteness": None}
    if use_animacy:
        clsf["animacy"] = BertClassifier(
            os.path.join(MODEL_PATH, "animacy_bert_model"),
            id2lab={0: "animate", 1: "inanimate"},
            max_length=max_length,
        )
    if use_nptype:
        clsf["nptype"] = BertClassifier(
            os.path.join(MODEL_PATH, "nptype_bert_model"),
            id2lab={0: "pronoun", 1: "common"},
            max_length=max_length,
        )
    if use_definiteness:
        clsf["definiteness"] = BertClassifier(
            os.path.join(MODEL_PATH, "definiteness_bert_model"),
            id2lab={0: "definite", 1: "indef"},
            max_length=max_length,
        )
    return clsf


# -------------------- Core prelabel routine --------------------
def prelabel_and_split_file(
    in_path: str,
    ok_out_path: str,
    invalid_out_path: str,
    progress_path: str,
    classifiers: Dict[str, Optional[BertClassifier]],
    batch_size: int = 64,
    debug: bool = False,
    debug_limit: int = 200,
):
    processed_lines = 0
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as pf:
                processed_lines = int(json.load(pf).get("processed", 0))
        except Exception:
            processed_lines = 0

    # 统计
    stats = {
        "lines_total": 0,
        "ok_lines": 0,
        "invalid_lines": 0,
        "subj_tagged": 0,
        "obj_tagged": 0,
    }
    label_counts = {"animacy": {}, "nptype": {}, "definiteness": {}}

    # 预数总行数用于进度条
    with open(in_path, "r", encoding="utf-8") as fin:
        stats["lines_total"] = sum(1 for _ in fin)

    ok_fout = open(ok_out_path, "a", encoding="utf-8")
    inv_fout = open(invalid_out_path, "a", encoding="utf-8")

    with open(in_path, "r", encoding="utf-8") as fin:
        pbar = tqdm(total=stats["lines_total"], desc=os.path.basename(in_path), initial=processed_lines)
        last_tick = time.time()

        for idx, line in enumerate(fin, 1):
            if idx <= processed_lines:
                continue
            if debug and (stats["ok_lines"] + stats["invalid_lines"]) >= debug_limit:
                break
            if not line.strip():
                continue

            record = json.loads(line)
            tokens = record.get("tokens") or []
            verbs = record.get("verbs") or []
            sent_text = " ".join(t.get("text", "") for t in tokens) if tokens else ""

            can_perturb = any(is_valid_structure(v) for v in verbs) if (tokens and verbs) else False

            if not can_perturb:
                inv_fout.write(line)
                stats["invalid_lines"] += 1
            else:
                jobs: Dict[str, List[Tuple[str, str, dict]]] = {"animacy": [], "nptype": [], "definiteness": []}
                for entry in verbs:
                    if not is_valid_structure(entry):
                        continue
                    subj = entry.get("subject")
                    obj = (entry.get("objects") or [None])[0]
                    for role, np_node in (("subject", subj), ("object", obj)):
                        if not np_node or not np_node.get("text"):
                            continue
                        np_text = np_node["text"]
                        if classifiers["animacy"]:
                            jobs["animacy"].append((sent_text, np_text, np_node))
                        if classifiers["nptype"]:
                            jobs["nptype"].append((sent_text, np_text, np_node))
                        if classifiers["definiteness"]:
                            jobs["definiteness"].append((sent_text, np_text, np_node))

                for key in ("animacy", "nptype", "definiteness"):
                    clf = classifiers.get(key)
                    task_jobs = jobs[key]
                    if clf and task_jobs:
                        pairs = [(s, n) for (s, n, _) in task_jobs]
                        preds = clf.predict_batch(pairs, batch_size=batch_size)
                        for pred_label, (_, _, target_node) in zip(preds, task_jobs):
                            target_node[key] = pred_label
                            label_counts[key][pred_label] = label_counts[key].get(pred_label, 0) + 1

                stats["subj_tagged"] += sum(1 for e in verbs if is_valid_structure(e) and e.get("subject"))
                stats["obj_tagged"] += sum(1 for e in verbs if is_valid_structure(e) and (e.get("objects") or [None])[0])

                ok_fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["ok_lines"] += 1

            # 定期保存进度
            now = time.time()
            if (now - last_tick) >= 60 or idx == stats["lines_total"]:
                with open(progress_path, "w", encoding="utf-8") as pf:
                    json.dump({"processed": idx}, pf)
                pbar.n = idx
                pbar.set_postfix({"ok": stats["ok_lines"], "invalid": stats["invalid_lines"]})
                pbar.refresh()
                last_tick = now

        pbar.close()

    ok_fout.close()
    inv_fout.close()

    summary = {
        "input": in_path,
        "ok_output": ok_out_path,
        "invalid_output": invalid_out_path,
        "progress_file": progress_path,
        "stats": stats,
        "label_counts": label_counts,
    }
    sum_path = ok_out_path.replace("_verbs_labeled.ok.jsonl", "_summary.json")
    with open(sum_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)
    print(f"[DONE] OK -> {ok_out_path} | INVALID -> {invalid_out_path}")
    print(f"[STATS] {summary}")


# -------------------- Main (fixed resume logic) --------------------
def main():
    ap = argparse.ArgumentParser(description="Prelabel NP features and split into perturbable vs invalid.")
    ap.add_argument("--input_dir", type=str, default=os.path.join(DATA_PATH, "structured"))
    ap.add_argument("--output_dir", type=str, default=os.path.join(DATA_PATH, "structured_labeled"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_limit", type=int, default=2000)
    ap.add_argument("--no_animacy", action="store_true")
    ap.add_argument("--no_nptype", action="store_true")
    ap.add_argument("--no_definiteness", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    classifiers = load_classifiers(
        use_animacy=not args.no_animacy,
        use_nptype=not args.no_nptype,
        use_definiteness=not args.no_definiteness,
        max_length=args.max_length,
    )

    inputs = sorted(glob(os.path.join(args.input_dir, "*_verbs.jsonl")))
    if not inputs:
        raise FileNotFoundError(f"No *_verbs.jsonl under {args.input_dir}")

    for ip in inputs:
        ok_out = os.path.join(args.output_dir, os.path.basename(ip).replace("_verbs.jsonl", "_verbs_labeled.ok.jsonl"))
        inv_out = os.path.join(args.output_dir, os.path.basename(ip).replace("_verbs.jsonl", "_verbs_labeled.invalid.jsonl"))
        prog = os.path.join(args.output_dir, os.path.basename(ip).replace("_verbs.jsonl", "_progress.json"))

        if args.overwrite:
            for p in (ok_out, inv_out, prog):
                if os.path.exists(p):
                    os.remove(p)
        else:
            # ✅ 修复：如果有 progress 文件，优先断点续跑，而不是跳过
            if os.path.exists(prog):
                print(f"[RESUME] Found progress file: {prog}. Resuming from checkpoint...")
            elif os.path.exists(ok_out) and os.path.exists(inv_out):
                print(f"[SKIP] Exists: {ok_out} & {inv_out}. Use --overwrite to rebuild.")
                continue

        prelabel_and_split_file(
            in_path=ip,
            ok_out_path=ok_out,
            invalid_out_path=inv_out,
            progress_path=prog,
            classifiers=classifiers,
            batch_size=args.batch_size,
            debug=args.debug,
            debug_limit=args.debug_limit,
        )

    print(f"[ALL DONE] Wrote labeled JSONL files to {args.output_dir}")


if __name__ == "__main__":
    main()
