#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_minimal_pairs_v4.py — valid+test 双阶段平衡版
====================================================
从 structured_labeled/{valid,test}_verbs_labeled.ok.jsonl 构造 minimal pairs。

规则：
- 一半 pair:   good=应标记 → 加标; bad=不加标
- 一半 pair:   good=不应标记 → 不加标; bad=强制加标
- minimal: 两句仅在一个 NP 的标记上不同
- 数据来源: valid + test
- 与 run_perturb_v2.py 参数体系一致

输出：
$EVALUATION_PATH/minimal_pairs/.../valid_test_minimal_pairs.jsonl
+ summary.json
"""

import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

from utils import DATA_PATH, EVALUATION_PATH, AGENT_MARK, PATIENT_MARK
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
        # 修正：假设 span 是闭区间
        i = e + 1
    while i < len(tokens):
        out.append(tokens[i]["text"])
        i += 1
    return " ".join(out)


def stripped(s):
    """移除格标后的纯文本"""
    return s.replace(AGENT_MARK, "").replace(PATIENT_MARK, "").replace("  ", " ").strip()


# ==============================
# 抽样函数
# ==============================

def collect_samples(input_paths, system, A_mode, P_mode, P_combo,
                    inverse, compare_attr, direction, target_type):
    """
    收集某一类样本：
    target_type ∈ {"should_mark", "should_not_mark"}
    input_paths: [valid_path, test_path]
    """
    collected = []
    for path in input_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                tokens = data.get("tokens") or []
                verbs = data.get("verbs") or []
                if not tokens or not verbs:
                    continue

                for entry in verbs:
                    if not is_valid_structure(entry):
                        continue
                    subj, objs = entry["subject"], entry["objects"]
                    if not objs or len(objs) != 1:
                        continue
                    obj = objs[0]
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

                    need_A, need_P = res.get("A_mark", False), res.get("P_mark", False)

                    # 根据目标类型筛选
                    if target_type == "should_mark" and not (need_A or need_P):
                        continue
                    if target_type == "should_not_mark" and (need_A or need_P):
                        continue

                    # 确定角色与标记符号
                    if need_P or (target_type == "should_not_mark" and not need_A):
                        role, mark, span = "P", PATIENT_MARK, dict(obj)
                    else:
                        role, mark, span = "A", AGENT_MARK, dict(subj)

                    span_marked = dict(span)
                    span_marked["text"] = f"{span['text']} {mark}"

                    marked_sent = apply_spans_to_tokens(tokens, [span_marked])
                    unmarked_sent = apply_spans_to_tokens(tokens, [])
                    if stripped(marked_sent) != stripped(unmarked_sent):
                        collected.append({
                            "role": role,
                            "marked": marked_sent,
                            "unmarked": unmarked_sent
                        })
    return collected


# ==============================
# 主逻辑
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
    ap.add_argument("--num_pairs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    valid_path = os.path.join(DATA_PATH, "structured_labeled", "valid_verbs_labeled.ok.jsonl")
    test_path = os.path.join(DATA_PATH, "structured_labeled", "test_verbs_labeled.ok.jsonl")
    input_paths = [valid_path, test_path]

    # 输出目录命名与 run_perturb_v2 对齐
    if args.system == "global":
        out_dir = os.path.join(EVALUATION_PATH, f"minimal_pairs/{args.system}_{args.compare_attr}_{args.direction}")
    elif args.system == "dualP":
        inv_tag = "_inv" if args.inverse else ""
        out_dir = os.path.join(EVALUATION_PATH, f"minimal_pairs/{args.system}_{args.P_combo}{inv_tag}")
    elif args.system in {"independent", "parallel"}:
        inv_tag = "_inv" if args.inverse else ""
        out_dir = os.path.join(EVALUATION_PATH, f"minimal_pairs/{args.system}_A{args.A_mode}_P{args.P_mode}{inv_tag}")
    else:
        raise ValueError(f"Unknown system type: {args.system}")

    os.makedirs(out_dir, exist_ok=True)
    half = args.num_pairs // 2

    print(f"[INFO] Collecting should-mark samples (valid+test)...")
    should_mark_samples = collect_samples(
        input_paths, args.system, args.A_mode, args.P_mode,
        args.P_combo, args.inverse, args.compare_attr, args.direction,
        "should_mark"
    )
    print(f"  found {len(should_mark_samples)} candidates")

    print(f"[INFO] Collecting should-NOT-mark samples (valid+test)...")
    should_not_mark_samples = collect_samples(
        input_paths, args.system, args.A_mode, args.P_mode,
        args.P_combo, args.inverse, args.compare_attr, args.direction,
        "should_not_mark"
    )
    print(f"  found {len(should_not_mark_samples)} candidates")

    random.shuffle(should_mark_samples)
    random.shuffle(should_not_mark_samples)
    picked_mark = should_mark_samples[:half]
    picked_unmark = should_not_mark_samples[:half]

    pairs = []
    for item in picked_mark:
        pairs.append({
            "sentence_good": item["marked"],
            "sentence_bad": item["unmarked"],
            "role": item["role"],
            "type": "should_mark"
        })
    for item in picked_unmark:
        pairs.append({
            "sentence_good": item["unmarked"],
            "sentence_bad": item["marked"],
            "role": item["role"],
            "type": "should_not_mark"
        })
    random.shuffle(pairs)

    out_path = Path(out_dir) / "valid_test_minimal_pairs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Generated {len(pairs)} pairs "
          f"({len(picked_mark)} should-mark, {len(picked_unmark)} should-not-mark)")
    print(f"[OUTPUT] {out_path}")

    summary = {
        "total_pairs": len(pairs),
        "should_mark": len(picked_mark),
        "should_not_mark": len(picked_unmark),
        "inputs": input_paths,
        "output": str(out_path)
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
