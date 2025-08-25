import os
import json
import argparse
from glob import glob
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils import DATA_PATH, MODEL_PATH, AGENT_MARK, PATIENT_MARK

# ---------------- Perf knobs (safe defaults) ----------------
# 减少多核调度开销；小 batch/频繁调用场景更稳
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

# ---------------- Config ----------------
MODEL_DIR = os.path.join(MODEL_PATH, "animacy_bert_model")
LABEL_MAP = {0: "human", 1: "animal", 2: "inanimate", 3: "event"}
ANIMACY_RANK = {"human": 3, "animal": 2, "inanimate": 1, "event": 0}

# ---------------- Model ----------------
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Animacy model not found: {MODEL_DIR}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 用 fast tokenizer（Rust 实现）显著加速分词
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# CUDA 下启用半精度推理（对 BERT 分类非常稳）
if device.type == "cuda":
    model.half()

# ---------------- Simple prediction cache ----------------
# 避免同一 (sentence, np_text) 被重复分类
pred_cache: dict[tuple[str, str], str] = {}

# ---------------- Helpers ----------------
def predict_animacy(sentence: str, np_text: str) -> str:
    key = (sentence, np_text)
    if key in pred_cache:
        return pred_cache[key]

    text = f"{sentence} [NP] {np_text}"
    # 纯推理：inference_mode 可避免 autograd 开销
    with torch.inference_mode():
        if device.type == "cuda":
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs).logits
        else:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits

    label = LABEL_MAP[int(torch.argmax(logits, dim=1).item())]
    pred_cache[key] = label
    return label

def compare_animacy(subj_cat: str, obj_cat: str) -> str:
    if ANIMACY_RANK[subj_cat] > ANIMACY_RANK[obj_cat]: return "higher"
    if ANIMACY_RANK[subj_cat] < ANIMACY_RANK[obj_cat]: return "lower"
    return "equal"

def decision_by_mode(mode: str, subj_cat: str, obj_cat: str) -> bool:
    if mode == "rule":       # 1) 基于比较（S ≤ O 才加）
        return compare_animacy(subj_cat, obj_cat) in {"lower", "equal"}
    if mode == "heuristic":  # 2) 只看主语（S==human 才加）
        return subj_cat == "human"
    if mode == "full":       # 3) 全部加
        return True
    if mode == "none":       # 4) 都不加
        return False
    raise ValueError(f"Unknown mode: {mode}")

def is_valid_structure(entry: dict) -> bool:
    return bool(entry.get("subject")) and len(entry.get("objects", [])) == 1

def apply_spans_to_tokens(tokens, spans):
    """按非重叠 [start,end] 替换，返回新句子文本。"""
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

def mark_for_strategy(strategy: str, subj_span: dict, obj_span: dict):
    spans = []
    if strategy in {"A+P", "A_only"}:
        s = dict(subj_span); s["text"] = f"{s['text']} {AGENT_MARK}"; s["_role"] = "A"; spans.append(s)
    if strategy in {"A+P", "P_only"}:
        o = dict(obj_span);  o["text"] = f"{o['text']} {PATIENT_MARK}"; o["_role"] = "P"; spans.append(o)
    # 去重：同一 (start,end,role) 仅保留一次，防止多动词重复标注
    uniq = {}
    for sp in spans:
        key = (sp["span"][0], sp["span"][1], sp.get("_role"))
        if key not in uniq:
            uniq[key] = sp
    return list(uniq.values())

# ---------------- Core per-file ----------------
def process_one_jsonl(input_path: str, mode: str, strategy: str, out_dir: str, written_list: list):
    fname = os.path.basename(input_path)
    prefix = fname.replace("_verbs.jsonl", "")
    out_affected   = os.path.join(out_dir, f"{prefix}_affected.txt")
    out_unaffected = os.path.join(out_dir, f"{prefix}_unaffected.txt")
    out_invalid    = os.path.join(out_dir, f"{prefix}_invalid.txt")

    affected_lines, unaffected_lines, invalid_lines = [], [], []

    # 统计：句子数与 token 数（用原始 tokens 长度）
    stats = {
        "split": prefix,
        "paths": {
            "affected": out_affected,
            "unaffected": out_unaffected,
            "invalid": out_invalid,
        },
        "counts": {
            "affected": {"sentences": 0, "tokens": 0},
            "unaffected": {"sentences": 0, "tokens": 0},
            "invalid": {"sentences": 0, "tokens": 0},
        },
        "mode": mode,
        "strategy": strategy,
        "source": input_path,
    }

    would_affect_sentence_count = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"{mode}-{strategy} :: {fname}"):
            if not line.strip(): continue
            data = json.loads(line)
            tokens = data.get("tokens") or []
            verbs  = data.get("verbs") or []
            sent_text = " ".join(t["text"] for t in tokens) if tokens else ""
            tok_len = len(tokens)

            has_valid = False
            has_affected = False
            would_affect_flag = False
            spans = []

            if not tokens or not verbs:
                invalid_lines.append(sent_text)
                stats["counts"]["invalid"]["sentences"] += 1
                stats["counts"]["invalid"]["tokens"] += tok_len
                continue

            for entry in verbs:
                if not is_valid_structure(entry):
                    continue
                has_valid = True
                subj = entry["subject"]
                obj  = entry["objects"][0]

                if mode in {"rule", "heuristic"}:
                    subj_cat = predict_animacy(sent_text, subj["text"])
                    obj_cat  = predict_animacy(sent_text, obj["text"])
                    decision = decision_by_mode(mode, subj_cat, obj_cat)
                else:
                    decision = decision_by_mode(mode, "", "")

                if decision:
                    if mode == "none":
                        would_affect_flag = True
                    else:
                        add_spans = mark_for_strategy(strategy, subj, obj)
                        if add_spans:
                            spans.extend(add_spans)
                            has_affected = True

            # 防止跨多个动词导致的重复 span
            if spans:
                uniq = {}
                for sp in spans:
                    key = (sp["span"][0], sp["span"][1], sp.get("_role"), sp["text"])
                    if key not in uniq:
                        uniq[key] = sp
                spans = list(uniq.values())

            if has_affected and spans:
                affected_lines.append(apply_spans_to_tokens(tokens, spans))
                stats["counts"]["affected"]["sentences"] += 1
                stats["counts"]["affected"]["tokens"] += tok_len
            elif has_valid:
                if mode == "none" and would_affect_flag:
                    would_affect_sentence_count += 1
                unaffected_lines.append(sent_text)
                stats["counts"]["unaffected"]["sentences"] += 1
                stats["counts"]["unaffected"]["tokens"] += tok_len
            else:
                invalid_lines.append(sent_text)
                stats["counts"]["invalid"]["sentences"] += 1
                stats["counts"]["invalid"]["tokens"] += tok_len

    os.makedirs(out_dir, exist_ok=True)
    with open(out_affected, "w", encoding="utf-8") as f:
        for s in affected_lines: f.write(s + "\n")
    with open(out_unaffected, "w", encoding="utf-8") as f:
        for s in unaffected_lines: f.write(s + "\n")
    with open(out_invalid, "w", encoding="utf-8") as f:
        for s in invalid_lines: f.write(s + "\n")

    written_list += [out_affected, out_unaffected, out_invalid]

    # 控制台摘要
    print(f"\n[{prefix}] Mode={mode} | Strategy={strategy}")
    print(f"  Affected   : {stats['counts']['affected']['sentences']}")
    print(f"  Unaffected : {stats['counts']['unaffected']['sentences']}")
    print(f"  Invalid    : {stats['counts']['invalid']['sentences']}")
    if mode == "none":
        print(f"  Would-affect (no-mark): {would_affect_sentence_count}")
    print(f"  -> {out_affected}\n  -> {out_unaffected}\n  -> {out_invalid}")

    return stats

# ---------------- Main ----------------
def main(mode: str, strategy: str):
    assert mode in {"rule", "heuristic", "full", "none"}
    assert strategy in {"A+P", "A_only", "P_only"}

    out_dir = os.path.join(DATA_PATH, f"perturbed_model/{mode}_{strategy}")
    structured_dir = os.path.join(DATA_PATH, "structured")
    inputs = sorted(glob(os.path.join(structured_dir, "*_verbs.jsonl")))
    if not inputs:
        raise FileNotFoundError(f"No *_verbs.jsonl under {structured_dir}")

    written = []
    all_stats = []
    total = {
        "affected": {"sentences": 0, "tokens": 0},
        "unaffected": {"sentences": 0, "tokens": 0},
        "invalid": {"sentences": 0, "tokens": 0},
    }

    for ip in inputs:
        stats = process_one_jsonl(ip, mode, strategy, out_dir, written)
        all_stats.append(stats)
        # 累加总计
        for k in ("affected", "unaffected", "invalid"):
            total[k]["sentences"] += stats["counts"][k]["sentences"]
            total[k]["tokens"]    += stats["counts"][k]["tokens"]

    # 写入汇总配置（第十个文件）
    summary_path = os.path.join(out_dir, f"summary_{mode}_{strategy}.json")
    summary = {
        "mode": mode,
        "strategy": strategy,
        "output_dir": out_dir,
        "files": all_stats,   # 每个 split 的详细计数与路径
        "total": total,       # 三个 split 的合计
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[SUMMARY] wrote {len(written)} text files + 1 summary:")
    for p in written: print(" -", p)
    print(" -", summary_path)

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="rule",
                    choices=["rule", "heuristic", "full", "none"])
    ap.add_argument("--strategy", type=str, default="A+P",
                    choices=["A+P", "A_only", "P_only"])
    args = ap.parse_args()
    main(args.mode, args.strategy)
