import os
import json
import argparse
from collections import defaultdict, Counter
from functools import lru_cache
from tqdm import tqdm
from pathlib import Path

import spacy
from benepar import BeneparComponent  # noqa: F401 (ensures model is available)
import benepar  # noqa: F401

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from utils import DATA_PATH, MODEL_PATH, AGENT_MARK, PATIENT_MARK, EVALUATION_PATH


# -------------------------
# spaCy + benepar
# -------------------------
_nlp = spacy.load("en_core_web_trf")
if "benepar" not in _nlp.pipe_names:
    _nlp.add_pipe("benepar", config={"model": "benepar_en3"}, last=True)


def spacy_annotate(sent: str):
    doc = _nlp(sent.strip())
    s = list(doc.sents)[0]
    tokens = [
        {
            "id": tok.i,
            "text": tok.text,
            "lemma": tok.lemma_,
            "pos": tok.pos_,
            "dep": tok.dep_,
            "head": tok.head.i,
            "ner": tok.ent_type_ or "O",
        }
        for tok in s
    ]
    return tokens


# -------------------------
# Verb structure extraction
# -------------------------
def build_children_index(tokens):
    children = defaultdict(list)
    for t in tokens:
        children[t["head"]].append(t)
    return children


def extract_np_span(token_id, id2tok, children):
    head = id2tok[token_id]
    span_tokens = [head]
    for child in children[token_id]:
        if child["dep"] in {"det", "amod", "compound", "poss", "nmod"}:
            span_tokens.append(child)
    span_tokens.sort(key=lambda x: x["id"])
    ids = [t["id"] for t in span_tokens]
    return {
        "text": " ".join(t["text"] for t in span_tokens),
        "span": [min(ids), max(ids)],
        "head": head["text"],
    }


def extract_clause_span(token_id, id2tok, children):
    visited = set()
    to_visit = [token_id]
    while to_visit:
        cur = to_visit.pop()
        visited.add(cur)
        for child in children[cur]:
            if child["id"] not in visited:
                to_visit.append(child["id"])
    span_tokens = sorted([id2tok[i] for i in visited], key=lambda x: x["id"])
    ids = [t["id"] for t in span_tokens]
    return {
        "text": " ".join(t["text"] for t in span_tokens),
        "span": [min(ids), max(ids)],
        "head": id2tok[token_id]["text"],
    }


def extract_verb_arguments(tokens):
    results = []
    id2tok = {t["id"]: t for t in tokens}
    children = build_children_index(tokens)
    verbs = [t for t in tokens if t["pos"] == "VERB"]

    for verb in verbs:
        vid = verb["id"]
        result = {
            "verb": verb["lemma"],
            "verb_id": vid,
            "subject": None,
            "objects": [],
        }
        for t in children[vid]:
            if t["dep"] in {"nsubj", "nsubjpass", "csubj"}:
                result["subject"] = extract_np_span(t["id"], id2tok, children)
            elif t["dep"] in {"dobj", "obj", "attr", "dative"}:
                result["objects"].append({**extract_np_span(t["id"], id2tok, children), "dep": t["dep"]})
            elif t["dep"] in {"xcomp", "ccomp"}:
                result["objects"].append({**extract_clause_span(t["id"], id2tok, children), "dep": t["dep"]})
            elif t["dep"] == "prep":
                prep = t
                subtree = [prep]
                for ch in children[prep["id"]]:
                    if ch["dep"] == "pobj":
                        subtree.append(ch)
                        for gc in children[ch["id"]]:
                            if gc["dep"] in {"det", "amod", "compound", "poss"}:
                                subtree.append(gc)
                if len(subtree) > 1:
                    ids = [tok["id"] for tok in subtree]
                    span = sorted(subtree, key=lambda x: x["id"])
                    result["objects"].append({
                        "text": " ".join(tok["text"] for tok in span),
                        "span": [min(ids), max(ids)],
                        "head": next((tok["text"] for tok in subtree if tok["dep"] == "pobj"), None),
                        "dep": "prep+pobj",
                    })
        results.append(result)
    return results


# -------------------------
# Animacy model and rules
# -------------------------
label_map = {0: "human", 1: "animal", 2: "inanimate", 3: "event"}
animacy_rank = {"human": 3, "animal": 2, "inanimate": 1, "event": 0}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_animacy_model_path = os.path.join(MODEL_PATH, "animacy_bert_model")
_animacy_tokenizer = BertTokenizer.from_pretrained(_animacy_model_path)
_animacy_model = BertForSequenceClassification.from_pretrained(_animacy_model_path).to(device).eval()


@lru_cache(maxsize=100000)
def predict_animacy(sentence: str, np_text: str):
    text = f"{sentence} [NP] {np_text}"
    inputs = _animacy_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _animacy_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred]


def compare_animacy(subj_cat, obj_cat):
    if animacy_rank[subj_cat] > animacy_rank[obj_cat]:
        return "higher"
    elif animacy_rank[subj_cat] < animacy_rank[obj_cat]:
        return "lower"
    else:
        return "equal"


def should_perturb_rule(subj_cat, obj_cat):
    return compare_animacy(subj_cat, obj_cat) in {"lower", "equal"}


def should_perturb_heuristic(subj_cat):
    return subj_cat == "human"


def is_valid_structure(entry):
    return entry.get("subject") and len(entry.get("objects", [])) == 1


def apply_spans_to_tokens(tokens, spans):
    span_map = {span["span"][0]: span["text"] for span in spans}
    skip_ids = set()
    for span in spans:
        start, end = span["span"]
        skip_ids.update(range(start + 1, end + 1))
    output = []
    for i, tok in enumerate(tokens):
        if i in skip_ids:
            continue
        elif i in span_map:
            output.append(span_map[i])
        else:
            output.append(tok["text"])
    return " ".join(output)


def perturb_one_sentence(sentence: str, mode: str, strategy: str):
    """
    Returns (processed_sentence, is_valid).
    If the sentence has no valid structure, returns (None, False).
    If it has valid structure but no rule hits, returns (original_sentence, True).
    If it has valid structure and rule hits, returns (marked_sentence, True).
    """
    tokens = spacy_annotate(sentence)
    if not tokens:
        return None, False

    verbs = extract_verb_arguments(tokens)
    if not verbs:
        return None, False

    any_valid = False
    spans = []
    sentence_text = " ".join(t["text"] for t in tokens)

    for entry in verbs:
        if not is_valid_structure(entry):
            continue
        any_valid = True

        subj = entry["subject"]
        obj = entry["objects"][0]
        if obj.get("dep") == "ccomp":
            continue

        subj_cat = predict_animacy(sentence_text, subj["text"])
        obj_cat = predict_animacy(sentence_text, obj["text"])

        if strategy == "full":
            do_mark = True
        else:
            do_mark = (
                (mode == "rule" and should_perturb_rule(subj_cat, obj_cat)) or
                (mode == "heuristic" and should_perturb_heuristic(subj_cat))
            )

        if not do_mark:
            continue

        if strategy in {"A+P", "A_only", "full"}:
            s = dict(subj)
            s["text"] += f" {AGENT_MARK}"
            spans.append(s)
        if strategy in {"A+P", "P_only", "full"}:
            o = dict(obj)
            o["text"] += f" {PATIENT_MARK}"
            spans.append(o)


    if not any_valid:
        return None, False

    if spans:
        return apply_spans_to_tokens(tokens, spans), True
    else:
        return sentence_text, True


def process_blimp(jsonl_path: str, out_path: str, mode: str, strategy: str, limit=None):
    kept = 0
    total = 0
    stats = Counter()

    with open(jsonl_path, encoding="utf-8") as fin:
        lines = [line for line in fin if line.strip()]
    total_lines = len(lines)

    with open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(lines, desc=f"Processing {os.path.basename(jsonl_path)}", total=total_lines):

            if not line.strip():
                continue
            total += 1
            if limit and kept >= limit:
                break

            item = json.loads(line)
            good = item.get("sentence_good", "")
            bad = item.get("sentence_bad", "")
            if not good or not bad:
                stats["missing_fields"] += 1
                continue

            good_out, good_valid = perturb_one_sentence(good, mode=mode, strategy=strategy)
            bad_out, bad_valid = perturb_one_sentence(bad, mode=mode, strategy=strategy)

            # Keep pairs where both sentences have valid structure.
            if good_valid and bad_valid:
                fout.write(json.dumps({
                    "sentence_good": good_out,
                    "sentence_bad": bad_out,
                    "orig_sentence_good": good,
                    "orig_sentence_bad": bad,
                    "UID": item.get("UID"),
                    "pairID": item.get("pairID"),
                    "field": item.get("field"),
                    "linguistics_term": item.get("linguistics_term")
                }, ensure_ascii=False) + "\n")
                kept += 1
            else:
                stats["filtered_pairs"] += 1

    print(f"\nMode={mode} | Strategy={strategy}")
    print(f"Input pairs: {total}")
    print(f"Kept pairs (both have valid structure): {kept}")
    print("Stats:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blimp", type=str, required=True, help="Path to BLiMP *.jsonl (with sentence_good/sentence_bad)")
    parser.add_argument("--out", type=str, required=True, help="Output path or directory for processed pairs")
    parser.add_argument("--mode", type=str, choices=["rule", "heuristic"], default="rule")
    parser.add_argument("--strategy", type=str, choices=["A+P", "A_only", "P_only", "full"], default="A+P")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    blimp_name = Path(args.blimp).stem
    suffix = f"{args.mode}_{args.strategy}"
    out_path = Path(args.out)

    if out_path.is_dir() or not out_path.suffix:
        out_path.mkdir(parents=True, exist_ok=True)
        final_out = out_path / f"{blimp_name}_{suffix}.jsonl"
    else:
        final_out = out_path.with_name(f"{out_path.stem}_{suffix}.jsonl")
        final_out.parent.mkdir(parents=True, exist_ok=True)

    process_blimp(args.blimp, str(final_out), mode=args.mode, strategy=args.strategy, limit=args.limit)

