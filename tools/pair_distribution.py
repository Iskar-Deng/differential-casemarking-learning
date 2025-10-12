#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计主语-宾语配对分布 + 主语/宾语单独分布
===========================================================
输入：*_verbs_labeled.ok.jsonl
输出：JSON 格式统计结果
"""

import os
import json
import argparse
from glob import glob
from collections import Counter
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing *_verbs_labeled.ok.jsonl")
    ap.add_argument("--output", default="pair_stats_v2.json",
                    help="Output JSON file path")
    args = ap.parse_args()

    files = sorted(glob(os.path.join(args.input_dir, "*_verbs_labeled.ok.jsonl")))
    if not files:
        raise FileNotFoundError(f"No *_verbs_labeled.ok.jsonl found in {args.input_dir}")

    # 初始化统计容器
    pair_stats = {
        "nptype_pairs": Counter(),
        "animacy_pairs": Counter(),
        "definiteness_pairs": Counter(),
        "subj_nptype": Counter(),
        "obj_nptype": Counter(),
        "subj_animacy": Counter(),
        "obj_animacy": Counter(),
        "subj_definiteness": Counter(),
        "obj_definiteness": Counter(),
        "total_pairs": 0,
    }

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=os.path.basename(fp)):
                if not line.strip():
                    continue
                data = json.loads(line)
                for verb in data.get("verbs", []):
                    subj = verb.get("subject")
                    objs = verb.get("objects") or []
                    if not subj or len(objs) != 1:
                        continue
                    obj = objs[0]

                    subj_np = subj.get("nptype", "unknown")
                    obj_np = obj.get("nptype", "unknown")
                    subj_an = subj.get("animacy", "unknown")
                    obj_an = obj.get("animacy", "unknown")
                    subj_def = subj.get("definiteness", "unknown")
                    obj_def = obj.get("definiteness", "unknown")

                    # pair
                    pair_stats["nptype_pairs"][f"{subj_np}–{obj_np}"] += 1
                    pair_stats["animacy_pairs"][f"{subj_an}–{obj_an}"] += 1
                    pair_stats["definiteness_pairs"][f"{subj_def}–{obj_def}"] += 1

                    # marginals
                    pair_stats["subj_nptype"][subj_np] += 1
                    pair_stats["obj_nptype"][obj_np] += 1
                    pair_stats["subj_animacy"][subj_an] += 1
                    pair_stats["obj_animacy"][obj_an] += 1
                    pair_stats["subj_definiteness"][subj_def] += 1
                    pair_stats["obj_definiteness"][obj_def] += 1

                    pair_stats["total_pairs"] += 1

    # 转为普通 dict
    result = {k: dict(v) if isinstance(v, Counter) else v for k, v in pair_stats.items()}

    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2, ensure_ascii=False)

    print(f"[DONE] Stats written to {args.output}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
