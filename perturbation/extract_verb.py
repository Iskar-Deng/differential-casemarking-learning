# perturbation/extract_verb.py

import os
import json
import sys
from collections import defaultdict
from tqdm import tqdm
from utils import DATA_PATH

def build_children_index(tokens):
    children = defaultdict(list)
    for t in tokens:
        children[t["head"]].append(t)
    return children

def extract_np_span(token_id, tokens, children):
    head = tokens[token_id]
    span_tokens = [head]
    for child in children[token_id]:
        if child["dep"] in {"det", "amod", "compound", "poss", "nmod"}:
            span_tokens.append(child)
    span_tokens.sort(key=lambda x: x["id"])
    ids = [t["id"] for t in span_tokens]
    return {
        "text": " ".join(t["text"] for t in span_tokens),
        "span": [min(ids), max(ids)],
        "head": head["text"]
    }

def extract_clause_span(token_id, tokens, children):
    visited = set()
    to_visit = [token_id]
    while to_visit:
        current = to_visit.pop()
        visited.add(current)
        for child in children[current]:
            if child["id"] not in visited:
                to_visit.append(child["id"])
    span_tokens = sorted([tokens[i] for i in visited], key=lambda x: x["id"])
    ids = [t["id"] for t in span_tokens]
    return {
        "text": " ".join(t["text"] for t in span_tokens),
        "span": [min(ids), max(ids)],
        "head": tokens[token_id]["text"]
    }

def extract_verb_arguments(tokens):
    results = []
    id_to_token = {t["id"]: t for t in tokens}
    children = build_children_index(tokens)
    verbs = [t for t in tokens if t["pos"] == "VERB"]

    for verb in verbs:
        verb_id = verb["id"]
        result = {
            "verb": verb["lemma"],
            "verb_id": verb_id,
            "subject": None,
            "objects": []
        }

        for t in children[verb_id]:
            if t["dep"] in {"nsubj", "nsubjpass", "csubj"}:
                result["subject"] = extract_np_span(t["id"], id_to_token, children)
            elif t["dep"] in {"dobj", "obj", "attr", "dative"}:
                result["objects"].append({
                    **extract_np_span(t["id"], id_to_token, children),
                    "dep": t["dep"]
                })
            elif t["dep"] in {"xcomp", "ccomp"}:
                result["objects"].append({
                    **extract_clause_span(t["id"], id_to_token, children),
                    "dep": t["dep"]
                })
            elif t["dep"] == "prep":
                prep_token = t
                subtree_tokens = [prep_token]
                for child in children[prep_token["id"]]:
                    if child["dep"] == "pobj":
                        subtree_tokens.append(child)
                        for grandchild in children[child["id"]]:
                            if grandchild["dep"] in {"det", "amod", "compound", "poss"}:
                                subtree_tokens.append(grandchild)
                if len(subtree_tokens) > 1:
                    ids = [tok["id"] for tok in subtree_tokens]
                    span = sorted(subtree_tokens, key=lambda x: x["id"])
                    result["objects"].append({
                        "text": " ".join(tok["text"] for tok in span),
                        "span": [min(ids), max(ids)],
                        "head": next((tok["text"] for tok in subtree_tokens if tok["dep"] == "pobj"), None),
                        "dep": "prep+pobj"
                    })

        results.append(result)

    return results

def process_all():
    parsed_dir = os.path.join(DATA_PATH, "parsed")
    structured_dir = os.path.join(DATA_PATH, "structured")
    os.makedirs(structured_dir, exist_ok=True)

    jsonl_files = [f for f in os.listdir(parsed_dir) if f.endswith(".jsonl")]
    if len(jsonl_files) != 1:
        print(f"Expected exactly one .jsonl in {parsed_dir}, found: {jsonl_files}", file=sys.stderr)
        sys.exit(1)

    input_path = os.path.join(parsed_dir, jsonl_files[0])
    output_path = os.path.join(structured_dir, jsonl_files[0].replace(".jsonl", "_verbs.jsonl"))

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Extracting verb structures"):
            if not line.strip():
                continue
            data = json.loads(line)
            tokens = data.get("tokens", [])
            index = data.get("index", -1)
            extracted = extract_verb_arguments(tokens)
            fout.write(json.dumps({
                "index": index,
                "tokens": tokens,
                "verbs": extracted
            }, ensure_ascii=False) + "\n")

    print(f"Verb structures saved to: {output_path}")

if __name__ == "__main__":
    process_all()
