#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_perturb_v2.py — 最终版（with precomputed invalids）
========================================================
支持四大扰动系统（independent / dualP / global / parallel）。
从 structured_labeled/ 读取 .ok.jsonl，
自动从 data/invalid_texts/ 读取预处理 invalid。

输出：
  affected / unaffected / invalid + summary.json + aggregate_stats.json
支持 debug 模式（打印若干扰动样本）

--------------------------------------
python -m perturbation.run_perturb_v2 \
  --system independent \
  --A_mode none \
  --P_mode animate \
  --debug \
  --debug_limit 50
"""

import os
import json
import argparse
from glob import glob
from tqdm import tqdm
from utils import DATA_PATH, AGENT_MARK, PATIENT_MARK
from perturbation.rules_v2 import apply_rule


# ==============================
# 工具函数
# ==============================

def is_valid_structure(entry: dict) -> bool:
    """检查句法结构是否符合要求"""
    return bool(entry.get("subject")) and len(entry.get("objects", [])) == 1


def apply_spans_to_tokens(tokens, spans):
    """将修改后的 span 写回 token 序列"""
    if not spans:
        return " ".join(t["text"] for t in tokens)
    spans = sorted(spans, key=lambda s: s["span"][0])
    out, i = [], 0
    for sp in spans:
        s, e = sp["span"]
        while i < s and i < len(tokens):
            out.append(tokens[i]["text"])
            i += 1
        out.append(sp["text"])
        i = max(i, e + 1)
    while i < len(tokens):
        out.append(tokens[i]["text"])
        i += 1
    return " ".join(out)


# ==============================
# 主处理逻辑
# ==============================

def process_one_jsonl(
    input_path, out_dir, system,
    A_mode, P_mode, P_combo, inverse,
    compare_attr, direction,
    debug, debug_limit
):
    fname = os.path.basename(input_path)
    prefix = fname.replace("_verbs_labeled.ok.jsonl", "")

    affected, unaffected, invalid_new = [], [], []
    debug_samples = []
    stats = {"affected": 0, "unaffected": 0, "invalid": 0}

    print(f"[INFO] Processing OK file: {fname}")
    pbar = tqdm(desc=f"{fname}", unit="lines")

    with open(input_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin, 1):
            if debug and i > debug_limit:
                break
            if not line.strip():
                continue

            data = json.loads(line)
            tokens = data.get("tokens") or []
            verbs = data.get("verbs") or []
            sent_text = " ".join(t["text"] for t in tokens) if tokens else ""

            if not tokens or not verbs:
                invalid_new.append(sent_text)
                stats["invalid"] += 1
                continue

            has_valid, has_marked, spans = False, False, []
            for entry in verbs:
                if not is_valid_structure(entry):
                    continue
                has_valid = True
                subj = entry["subject"]
                obj = entry["objects"][0]
                if obj.get("dep") == "ccomp":
                    continue

                subj_labels = {
                    "animacy": subj.get("animacy"),
                    "nptype": subj.get("nptype"),
                    "definiteness": subj.get("definiteness"),
                }
                obj_labels = {
                    "animacy": obj.get("animacy"),
                    "nptype": obj.get("nptype"),
                    "definiteness": obj.get("definiteness"),
                }

                res = apply_rule(
                    system=system,
                    A_labels=subj_labels,
                    P_labels=obj_labels,
                    A_mode=A_mode,
                    P_mode=P_mode,
                    inverse=inverse,
                    feature=compare_attr,
                    combo=P_combo,
                    direction=direction,
                )

                if res.get("A_mark"):
                    s = dict(subj)
                    s["text"] = f"{s['text']} {AGENT_MARK}"
                    spans.append(s)
                    has_marked = True
                if res.get("P_mark"):
                    o = dict(obj)
                    o["text"] = f"{o['text']} {PATIENT_MARK}"
                    spans.append(o)
                    has_marked = True

            if has_valid and has_marked and spans:
                perturbed = apply_spans_to_tokens(tokens, spans)
                affected.append(perturbed)
                stats["affected"] += 1
                if debug and len(debug_samples) < 20:
                    debug_samples.append({
                        "original": sent_text,
                        "perturbed": perturbed,
                        "system": system,
                        "A_mode": A_mode,
                        "P_mode": P_mode,
                        "inverse": inverse
                    })
                    print(f"[DEBUG] {len(debug_samples)}: {sent_text} → {perturbed}")
            elif has_valid:
                unaffected.append(sent_text)
                stats["unaffected"] += 1
            else:
                invalid_new.append(sent_text)
                stats["invalid"] += 1

            if i % 10000 == 0:
                pbar.update(10000)
        pbar.update(i % 10000)
    pbar.close()

    # ==== 拼接已有 invalid ====
    split = "train" if "train" in fname else ("valid" if "valid" in fname else "test")
    pre_invalid_path = os.path.join(DATA_PATH, "invalid_texts", f"{split}_invalid.txt")
    all_invalid = []
    if os.path.exists(pre_invalid_path):
        print(f"[INFO] Loading precomputed invalids: {pre_invalid_path}")
        with open(pre_invalid_path, "r", encoding="utf-8") as finv:
            all_invalid = [line.strip() for line in finv if line.strip()]
    all_invalid.extend(invalid_new)

    # ==== 写出文件 ====
    with open(os.path.join(out_dir, f"{prefix}_affected.txt"), "w", encoding="utf-8") as f:
        for s in affected:
            f.write(s + "\n")
    with open(os.path.join(out_dir, f"{prefix}_unaffected.txt"), "w", encoding="utf-8") as f:
        for s in unaffected:
            f.write(s + "\n")
    with open(os.path.join(out_dir, f"{prefix}_invalid.txt"), "w", encoding="utf-8") as f:
        for s in all_invalid:
            f.write(s + "\n")

    # ==== debug 预览 ====
    if debug and debug_samples:
        dbg_path = os.path.join(out_dir, f"{prefix}_debug_preview.jsonl")
        with open(dbg_path, "w", encoding="utf-8") as f:
            for x in debug_samples:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[DEBUG] preview written to {dbg_path}")
        tqdm._instances.clear()

    return {"input": input_path, "stats": stats}


# ==============================
# 主程序入口
# ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["independent", "dualP", "global", "parallel"], required=True)
    ap.add_argument("--A_mode", type=str, default="none")
    ap.add_argument("--P_mode", type=str, default="none")
    ap.add_argument("--P_combo", choices=["and", "or"], default="and")
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--compare_attr", choices=["animacy", "nptype", "definiteness"], default="animacy")
    ap.add_argument("--direction", choices=["up", "down"], default="up")
    ap.add_argument("--input_dir", default=os.path.join(DATA_PATH, "structured_labeled"))
    ap.add_argument("--output_dir", default=os.path.join(DATA_PATH, "perturbed"))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_limit", type=int, default=100)
    args = ap.parse_args()

    inputs = sorted(glob(os.path.join(args.input_dir, "*_verbs_labeled.ok.jsonl")))
    if not inputs:
        raise FileNotFoundError(f"No *_verbs_labeled.ok.jsonl under {args.input_dir}")

    # === 输出目录命名逻辑 ===
    if args.system == "global":
        # global 系统命名示例: global_animacy_up
        out_dir = os.path.join(
            args.output_dir,
            f"{args.system}_{args.compare_attr}_{args.direction}"
        )
    elif args.system == "dualP":
        # dualP 系统命名示例: dualP_and_inv
        inv_tag = "_inv" if args.inverse else ""
        out_dir = os.path.join(
            args.output_dir,
            f"{args.system}_{args.P_combo}{inv_tag}"
        )
    elif args.system == "independent":
        # independent 系统命名示例: independent_Aanimate_Pnone_inv
        inv_tag = "_inv" if args.inverse else ""
        out_dir = os.path.join(
            args.output_dir,
            f"{args.system}_A{args.A_mode}_P{args.P_mode}{inv_tag}"
        )
    elif args.system == "parallel":
        # parallel 系统命名示例: parallel_Apronoun_Ppronoun
        inv_tag = "_inv" if args.inverse else ""
        out_dir = os.path.join(
            args.output_dir,
            f"{args.system}_A{args.A_mode}_P{args.P_mode}{inv_tag}"
        )
    else:
        raise ValueError(f"Unknown system type: {args.system}")

    os.makedirs(out_dir, exist_ok=True)

    summaries = []
    for ip in inputs:
        summaries.append(process_one_jsonl(
            ip, out_dir, args.system,
            args.A_mode, args.P_mode, args.P_combo,
            args.inverse, args.compare_attr, args.direction,
            args.debug, args.debug_limit
        ))

    # === 保存 summary 与 aggregate ===
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    agg_stats = {
        "system": args.system,
        "A_mode": args.A_mode,
        "P_mode": args.P_mode,
        "inverse": args.inverse,
        "compare_attr": args.compare_attr,
        "direction": args.direction,
        "files": len(inputs),
        "totals": {
            "affected": sum(x["stats"]["affected"] for x in summaries),
            "unaffected": sum(x["stats"]["unaffected"] for x in summaries),
            "invalid": sum(x["stats"]["invalid"] for x in summaries),
        },
    }
    agg_path = os.path.join(out_dir, "aggregate_stats.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg_stats, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Wrote outputs to {out_dir}")
    print(f" - summary.json")
    print(f" - aggregate_stats.json")


if __name__ == "__main__":
    main()
